import os
import cv2
import torch
import pickle
import argparse
import numpy as np
import warnings
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from detectron2.data.detection_utils import read_image
from supervision import Detections, BoxAnnotator, MaskAnnotator, LabelAnnotator, mask_to_xyxy

from sam2.build_sam import build_sam2_video_predictor
from VLPart.build_vlpart import build_vlpart_model


warnings.filterwarnings('ignore')

# Constants
SAM2_CONFIG = "sam2_hiera_l.yaml"
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
OUTPUT_ROOT = "/data/robot-merlin/mask_vlpart+sam2_tracking"
OUTPUT_ROOT_IMG = "/data/robot-merlin/mask_vlpart+sam2_tracking_with_image"

# Set up torch environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_affordance_data(pkl_path):
    """
    Load affordance data from a pickle file and organize it by video directory.
    Args:
        pkl_path (str): Path to the pickle file containing affordance data.
    Returns:
        dict: A dictionary where keys are video directory paths and values are lists of data entries.
    """
    with open(pkl_path, 'rb') as f:
        datas = pickle.load(f)

    data_dict = {}
    for data in datas:
        vid_path = os.path.dirname(data['frame_path'])
        data_dict.setdefault(vid_path, []).append(data)
    return data_dict


def init_vlpart_once(text, prev_text, vlpart_model):
    """
    Initialize VLPart model if the text has changed.
    """
    if text != prev_text:
        if vlpart_model is not None:
            del vlpart_model
        vlpart_model = build_vlpart_model(text)
    return vlpart_model, text


def run_vlpart_on_first_frame(vlpart_model, image_path):
    """
    Run VLPart model on the first frame to get bounding boxes.
    """
    img = read_image(image_path, format="BGR")
    predictions, _ = vlpart_model.run_on_image(img)
    if len(predictions["instances"]) != 1:
        return None
    return predictions["instances"].pred_boxes.tensor.cpu().numpy()


def run_sam2_tracking(video_dir, frame_names, sam2_predictor, boxes):
    """
    Run SAM2 tracking on the video frames using the provided bounding boxes.
    """
    inference_state = sam2_predictor.init_state(video_path=video_dir)
    sam2_predictor.reset_state(inference_state)

    _, obj_ids, mask_logits = sam2_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=boxes,
    )

    results = {}
    for frame_idx, out_ids, out_logits in sam2_predictor.propagate_in_video(inference_state):
        results[frame_idx] = {
            oid: (out_logits[i] > 0).cpu().numpy()
            for i, oid in enumerate(out_ids)
        }
    return results


def save_tracking_results(video_dir, frame_names, video_segments, object_name, output_base, vid):
    """
    Save the tracking results to the specified output directory.
    """
    objects = [object_name]
    id_to_objects = {i: obj for i, obj in enumerate(objects, start=1)}

    output_dir = Path(f"{output_base}/{vid:06d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_img = Path(f"{OUTPUT_ROOT_IMG}/{vid:06d}")
    output_dir_img.mkdir(parents=True, exist_ok=True)

    box_annotator = BoxAnnotator()
    label_annotator = LabelAnnotator()
    mask_annotator = MaskAnnotator()

    for idx, masks in video_segments.items():
        frame_path = os.path.join(video_dir, frame_names[idx])
        frame = cv2.imread(frame_path)

        obj_ids = list(masks.keys())
        mask_arr = np.concatenate(list(masks.values()), axis=0)

        detections = Detections(
            xyxy=mask_to_xyxy(mask_arr),
            mask=mask_arr,
            class_id=np.array(obj_ids, dtype=np.int32),
        )

        annotated = box_annotator.annotate(frame.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, [id_to_objects[i] for i in obj_ids])
        annotated = mask_annotator.annotate(annotated, detections)

        cv2.imwrite(str(output_dir_img / frame_names[idx]), annotated)
        cv2.imwrite(str(output_dir / frame_names[idx]), mask_arr[0] * 255)


def get_sorted_frame_names(video_dir):
    return sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ], key=lambda name: int(os.path.splitext(name)[0]))


def main(openx_data, text_override=None):
    # You can reorganize the data loading logic as needed
    data_dict = load_affordance_data(f'./data/{openx_data}_for_affordance.pkl')

    # Initialize SAM2 predictor
    sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)

    prev_text = ''
    vlpart_model = None

    for video_dir, data_list in tqdm(data_dict.items()):
        first_sample = data_list[0]
        frame_path = first_sample['frame_path']
        task_class = first_sample['task_object_class']

        # Only process specific classes
        if not any(k in task_class for k in ['door', 'drawer', 'knife']):
            continue

        # Initialize VLPart model with the task class
        input_text = f"{task_class} handle" if not text_override else text_override
        vlpart_model, prev_text = init_vlpart_once(input_text, prev_text, vlpart_model)

        # Process the first frame to get bounding boxes
        boxes = run_vlpart_on_first_frame(vlpart_model, frame_path)
        if boxes is None:
            continue

        # Run SAM2 tracking on the video frames
        frame_names = get_sorted_frame_names(video_dir)
        segments = run_sam2_tracking(video_dir, frame_names, sam2_predictor, boxes)
        save_tracking_results(video_dir, frame_names, segments, input_text,
                              f"{OUTPUT_ROOT}/", first_sample['vid'])
        print(f"[Done] {frame_path} | {task_class}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VLPart + SAM2 Tracking Demo")
    parser.add_argument("--pipeline", type=str, default="referring_expression_segmentation", help="Pipeline task")
    parser.add_argument("--text_input", type=str, default=None, help="Optional override for input text")
    parser.add_argument("--dataset", type=str, default="bridge", help="Dataset name (e.g., bridge)")
    args = parser.parse_args()

    main(args.dataset, args.pipeline, args.text_input)
