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