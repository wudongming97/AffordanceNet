## Dataset

To train our affordance segmentation model, we use two types of data:
* **General Segmentation Data**: This follows [LISA](https://github.com/dvlab-research/LISA).
* **Affordance Segmentation Data**: This is a large-scale dataset that we collect.

### Affordance Segmentation Data

We employ images from HANDAL, Open-X, GraspNet, EgoObjects, and RLBench in our affordance segmentation task. 

The HANDAL data is downloaded and organized according to its official [repo](https://github.com/NVlabs/HANDAL).
Other data can be downloaded from the [Hugging Face](https://huggingface.co/datasets/Dongming97/RAGNet).

The training data is organized as follows:
```
./data/
├── openx_train.pkl
├── graspnet_train.pkl
├── egoobjects_train.pkl
├── rlbench_train.pkl
├── handal_hard_reasoning_train.pkl
├── egoobjects_easy_reasoning_train.pkl
├── egoobjects_hard_reasoning_train.pkl
├── HANDAL
│   ├── without_depth
│       ├── handal_dataset_adjustable_wrenches
│       ├── handal_dataset_combinational_wrenches
│       ├── handal_dataset_fixed_joint_pliers
│       ├── ...
├── openx
│   ├── images
│       ├── fractal20220817_data
│       ├── bridge
│   ├── masks
│       ├── fractal20220817_data
│       ├── bridge
├── graspnet
│   ├── images
│   ├── masks
│   ├── test_seen
│   ├── test_novel
├── egoobjects
│   ├── images
│   ├── masks
├── rlbench
│   ├── images
│   ├── masks
├── 3doi
│   ├── images
│   ├── masks
```

The evaluation data is also in the same dictory, but with the `*_eval.pkl` files instead of `*_train.pkl`.

```
./data/
├── handal_mini_val.pkl
├── graspnet_test_seen_val.pkl
├── graspnet_test_novel_val.pkl
├── 3doi_val.pkl
├── handal_easy_reasoning_val.pkl
├── handal_hard_reasoning_val.pkl
├── 3doi_easy_reasoning_val.pkl
```

You can use the following script to confirm if data is organized correctly:
```bash
python data_curation/check_dataset.py
```

### About data curation
1. **SAM2**: We use SAM2 to generate affordance mask if the dataset provides box annotation.
2. **Florence-2 + SAM2**: We use Florence-2 to generate the initial segmentation masks of some complete objects, and then refine them with SAM2. Please see [Florence-2+SAM2](https://github.com/IDEA-Research/Grounded-SAM-2).
3. **VLPart + SAM2**: We use VLPart to generate box of object part, and then refine them with SAM2. We refer to [VLPart](https://github.com/facebookresearch/VLPart). 
We provide our inference demo scripts in `data_curation/build_vlpart.py` and `data_curation/vlpart_sam2_tracking.py`.
4. **Reasoning Instruction**: We provide two example scripts to generate reasoning instructions for the affordance segmentation task:
   - `data_curation/prompt_generation_handal_easy_reasoning.py`: This script generates easy reasoning instructions for the HANDAL dataset.
   - `data_curation/prompt_generation_handal_hard_reasoning.py`: This script generates hard reasoning instructions for the HANDAL dataset.