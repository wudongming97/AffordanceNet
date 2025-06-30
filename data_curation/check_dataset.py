import os
import pickle as pkl

DATA_DIR = './data'


def get_data_paths():
    """Retrieve train/val/reasoning/non-reasoning pkl file paths."""
    all_files = os.listdir(DATA_DIR)
    train_paths = [os.path.join(DATA_DIR, f) for f in all_files if f.endswith('train.pkl')]
    val_paths = [os.path.join(DATA_DIR, f) for f in all_files if f.endswith('val.pkl')]
    reasoning_paths = [os.path.join(DATA_DIR, f) for f in all_files if f.endswith('reasoning_val.pkl')]
    non_reasoning_paths = [vp for vp in val_paths if vp not in reasoning_paths]

    return train_paths, reasoning_paths, non_reasoning_paths


def check_file_exists(file_path, description=""):
    """Assert that the file exists, otherwise raise an error."""
    assert os.path.exists(file_path), f"{description} does not exist: {file_path}"


def check_train_data(train_path):
    """Check frame and mask paths for each sample in training data."""
    print(f"[Train] Checking: {train_path}")
    with open(train_path, "rb") as f:
        data = pkl.load(f)

    for item in data:
        check_file_exists(item["frame_path"], "Frame path")
        check_file_exists(item["mask_path"], "Mask path")

    print(f"[Train] ✅ Checked {train_path}. Samples: {len(data)}")


def check_val_data(val_path, reasoning=False):
    """Check validation data paths depending on reasoning mode."""
    tag = "Reasoning Val" if reasoning else "Non-Reasoning Val"
    print(f"[{tag}] Checking: {val_path}")

    with open(val_path, "rb") as f:
        data = pkl.load(f)

    if reasoning:
        for item in data:
            check_file_exists(item["frame_path"], "Frame path")
            check_file_exists(item["mask_path"], "Mask path")
        print(f"[{tag}] ✅ Checked {val_path}. Samples: {len(data)}")
    else:
        total_images = 0
        for class_name, image_list in data.get('images', {}).items():
            for image_path in image_list:
                check_file_exists(image_path, "Image path")
            total_images += len(image_list)

        for class_name, label_list in data.get('labels', {}).items():
            for label_path in label_list:
                check_file_exists(label_path, "Label path")

        print(f"[{tag}] ✅ Checked {val_path}. Samples: {total_images}")


def main():
    train_paths, reasoning_paths, non_reasoning_paths = get_data_paths()

    for train_path in train_paths:
        check_train_data(train_path)

    for val_path in non_reasoning_paths:
        check_val_data(val_path, reasoning=False)

    for val_path in reasoning_paths:
        check_val_data(val_path, reasoning=True)


if __name__ == "__main__":
    main()
