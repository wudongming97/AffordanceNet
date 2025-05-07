import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)
from PIL import Image

import pickle


AFFORDANCE_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "You are an embodied robot. Can you segment the affordance map of {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "You are an embodied robot. Please segment the affordance map of {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "You are an embodied robot. What is the affordance map of {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "You are an embodied robot. What is the affordance map of {class_name} in this image?",
]

AFFORDANCE_ANSWER_LIST = [
    "It is [AFF].",
    "Sure, [AFF].",
    "Sure, it is [AFF].",
    "Sure, the affordance map is [AFF].",
    "[AFF].",
]


class AffordanceSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        aff_seg_data="handal||openx||egoobjects",
        aff_sample_ratio=[1, 1, 1],
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.aff_seg_data = aff_seg_data
        aff_sample_ratio = np.array(aff_sample_ratio)
        self.aff_sample_ratio = aff_sample_ratio / aff_sample_ratio.sum()
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.affordance_question_list = AFFORDANCE_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        # self.answer_list = ANSWER_LIST
        self.answer_list = AFFORDANCE_ANSWER_LIST

        aff_seg_datas = aff_seg_data.split("||")
        self.data2list = {}
        self.object_ids = {}
        for ds in aff_seg_datas:
            if ds == "handal":
                aff_cls_list = os.listdir(os.path.join(base_image_dir, "HANDAL", "without_depth"))
                aff_cls_list = [aff_cls[15:] for aff_cls in aff_cls_list if '.zip' not in aff_cls]
                aff_cls_list = [aff_cls.replace('_', ' ') for aff_cls in aff_cls_list if len(aff_cls) > 0]
                images = {}
                labels = {}
                num_handal = 0
                for aff_cls in aff_cls_list:
                    images[aff_cls] = glob.glob(
                        os.path.join(
                            base_image_dir, "HANDAL", "without_depth",
                            'handal_dataset' + '_' + aff_cls.replace(' ', '_'),
                            'train', '*', 'rgb', '*.jpg'
                        )
                    )
                    labels[aff_cls] = [img.replace('rgb', 'mask_parts')[:-4] + '_000000_handle.png' for img in
                                      images[aff_cls]]
                    # masks[aff_cls] = [mask for mask in masks[aff_cls] if os.path.exists(mask)]
                    assert len(images[aff_cls]) == len(labels[aff_cls])
                    num_handal += len(images[aff_cls])
                self.data2list[ds] = (images, labels)
                print("categories of handal: ", aff_cls_list)
                print("number of handal samples: ", num_handal)
            elif ds == "openx" or ds == "egoobjects" or ds == "rlbench" or ds == "rlbenchv4":
                if ds == "rlbenchv4":
                    pkl_path = f'/data/robot-merlin/my_data/rlbench_data_v4.pkl'
                else:
                    pkl_path = f'/data/robot-merlin/my_data/{ds}_data.pkl'
                images = {}
                labels = {}
                with open(pkl_path, 'rb') as f:
                    aff_datas = pickle.load(f)
                for aff_data in aff_datas:
                    if aff_data['task_object_class'] not in images:
                        images[aff_data['task_object_class']] = []
                        labels[aff_data['task_object_class']] = []
                    images[aff_data['task_object_class']].append(aff_data['frame_path'])
                    labels[aff_data['task_object_class']].append(aff_data['mask_path'])
                # keep same numbers of samples for each class
                for k in images.keys():
                    assert len(images[k]) == len(labels[k])
                self.data2list[ds] = (images, labels)
                print(f"categories of {ds}: ", images.keys())
                print(f"number of {ds} samples: ", len(aff_datas))
            elif ds == 'graspnet':
                pkl_path = '/data/robot-merlin/my_data/graspnet_data.pkl'
                images = {}
                labels = {}
                object_ids = {}
                with open(pkl_path, 'rb') as f:
                    graspnet_datas = pickle.load(f)
                for graspnet_data in graspnet_datas:
                    if graspnet_data['task_object_class'] not in images:
                        images[graspnet_data['task_object_class']] = []
                        labels[graspnet_data['task_object_class']] = []
                        object_ids[graspnet_data['task_object_class']] = []
                    images[graspnet_data['task_object_class']].append(graspnet_data['frame_path'])
                    labels[graspnet_data['task_object_class']].append(graspnet_data['mask_path'])
                    if 'graspnet_object_id' in graspnet_data.keys():
                        object_ids[graspnet_data['task_object_class']].append(graspnet_data['graspnet_object_id'])
                    else:
                        object_ids[graspnet_data['task_object_class']].append(None)
                # keep same numbers of samples for each class
                for k in images.keys():
                    assert len(images[k]) == len(labels[k])
                    assert len(images[k]) == len(object_ids[k])
                self.data2list[ds] = (images, labels)
                self.object_ids[ds] = object_ids
                print(f"categories of {ds}: ", images.keys())
                print("number of graspnet samples: ", len(graspnet_datas))
            else:
                raise ValueError(f"Unsupported affordance segmentation dataset: {ds}")

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = np.random.choice(list(self.data2list.keys()), p=self.aff_sample_ratio)

        images, labels = self.data2list[ds]
        class_name = random.choice(list(images.keys()))
        idx = random.randint(0, len(images[class_name]) - 1)
        image_path = images[class_name][idx]
        label_path = labels[class_name][idx]
        if "rlbench" in ds:
            if "target" in class_name or "jar" in class_name or "button" in class_name:
                is_flip = random.random() > 0.5
                flip_code = random.choice([-1, 0, 1])
            elif "drawer" in class_name:
                is_flip = random.random() > 0.5
                flip_code = 1
            else:
                is_flip = False
                flip_code = 0
        else:
            is_flip = False
            flip_code = 0

        # load image and prepare input for clip and sam
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_flip:
            image = cv2.flip(image, flip_code)
        ori_size = image.shape[:2]
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # load class names
        sampled_classes = [class_name]

        # load label
        label = Image.open(label_path)
        label = np.array(label)
        if is_flip:
            label = cv2.flip(label, flip_code)
        label = torch.from_numpy(label).long()
        masks = []
        if ds == 'graspnet':
            object_id = self.object_ids[ds][class_name][idx]
            # if data is from graspnet and object_id exists, use the mask of the object_id
            if object_id is None:
                for _ in range(len(sampled_classes)):
                    masks.append(label > 0)
            else:
                for _ in range(len(sampled_classes)):
                    masks.append(label == object_id)
        else:
            for _ in range(len(sampled_classes)):
                masks.append(label > 0)
        masks = torch.stack(masks, dim=0)

        questions = []
        answers = []
        for sampled_cls in sampled_classes:
            text = sampled_cls

            assert len(text.split("||")) == 1
            question_template = random.choice(self.affordance_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )


class AffValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir.replace("/lisa_data", "")
        # splits = val_dataset.split("|")
        # ds, split = splits
        ds = val_dataset

        self.class_names = []
        self.class_ids = []
        self.images = []
        self.labels = []
        pkl_path = f'/data/robot-merlin/my_data/val_{ds}.pkl'
        if ds == 'handal_all':
            aff_cls_list = os.listdir(os.path.join(self.base_image_dir, "HANDAL", "without_depth"))
            aff_cls_list = [aff_cls[15:] for aff_cls in aff_cls_list if '.zip' not in aff_cls]
            aff_cls_list = [aff_cls.replace('_', ' ') for aff_cls in aff_cls_list if len(aff_cls) > 0]

            num_handal = 0
            images = {}
            labels = {}
            class_names = {}
            for aff_cls in aff_cls_list:
                images[aff_cls] = glob.glob(
                    os.path.join(
                        self.base_image_dir, "HANDAL", "without_depth",
                        'handal_dataset' + '_' + aff_cls.replace(' ', '_'),
                        'test', '*', 'rgb', '*.jpg'
                    )
                )
                labels[aff_cls] = [img.replace('rgb', 'mask_parts')[:-4] + '_000000_handle.png' for img in
                                   images[aff_cls]]
                class_names[aff_cls] = [aff_cls] * len(images[aff_cls])
                # masks[aff_cls] = [mask for mask in masks[aff_cls] if os.path.exists(mask)]
                assert len(images[aff_cls]) == len(labels[aff_cls])
                assert len(images[aff_cls]) == len(class_names[aff_cls])
                num_handal += len(images[aff_cls])

            for aff_cls in images.keys():
                self.images.extend(images[aff_cls])
                self.labels.extend(labels[aff_cls])
                self.class_names.extend(class_names[aff_cls])
                self.class_ids.extend([None] * len(images[aff_cls]))
            print(f'handal_all test number: {num_handal}')

        else:
            with open(pkl_path, 'rb') as f:
                val_datas = pickle.load(f)
            for class_name in val_datas['images'].keys():

                self.images.extend(val_datas['images'][class_name])
                self.labels.extend(val_datas['labels'][class_name])
                self.class_names.extend(val_datas['class_names'][class_name])
                if 'class_ids' in val_datas.keys():
                    self.class_ids.extend(val_datas['class_ids'][class_name])
                else:
                    self.class_ids.extend([None] * len(val_datas['images'][class_name]))

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):

        # load image
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # load class names
        sampled_sents = [self.class_names[idx]]

        # load label
        label_path = self.labels[idx]
        label = Image.open(label_path)
        label = np.array(label)
        label = torch.from_numpy(label).long()
        masks = []
        class_id = self.class_ids[idx]
        # if data object_id exists, use the mask of the object_id
        if class_id is None:
            for _ in range(len(sampled_sents)):
                masks.append(label > 0)
        else:
            for _ in range(len(sampled_sents)):
                masks.append(label == class_id)
        masks = torch.stack(masks, dim=0)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()

            conv.append_message(
                conv.roles[0],
                DEFAULT_IMAGE_TOKEN
                + "\nYou are an embodied robot. What is the affordance map of {} in this image?".format(
                    text
                ),
            )
            conv.append_message(conv.roles[1], "[AFF].")
            conversations.append(conv.get_prompt())
            i += 1

        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            None,
            None,
            inference,
        )