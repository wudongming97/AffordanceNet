<div align="center">
<h1>
<b>
RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping
</b>
</h1>
</div>

<div align="center">

| [**📑 Paper**](https://arxiv.org/abs/2507.23734)  |  [**🤗 Model**](https://huggingface.co/Dongming97/AffordanceVLM) |   [**🤗 Dataset**](https://huggingface.co/datasets/Dongming97/RAGNet) |  [**🖥️ Website**](https://wudongming97.github.io/RAGNet/) | 

</div>


<p align="center"><img src="./imgs/AffordanceNet.jpg" width="800"/></p>


> **[RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping](https://arxiv.org/abs/2507.23734)**
>
> Dongming Wu, Yanping Fu, Saike Huang, Yingfei Liu, Fan Jia, Nian Liu, Feng Dai, Tiancai Wang, Rao Muhammad Anwer, Fahad Shahbaz Khan, Jianbing Shen

## 📝 TL;DR
- To push forward general robotic grasping, we introduce a large-scale reasoning-based affordance segmentation benchmark, **RAGNet**.  It contains 273k images, 180 categories, and 26k reasoning instructions. 
- Furthermore, we propose a comprehensive affordance-based grasping framework, named AffordanceNet, which consists of a VLM (named AffordanceVLM) pre-trained on our massive affordance data and a grasping network that conditions an affordance map to grasp the target.

---

## 📰 News
- [2025.08] Paper is released at [arXiv](https://arxiv.org/abs/2507.23734).
- [2025.07] Inference code and the [AffordanceVLM](https://huggingface.co/Dongming97/AffordanceVLM) model are released. Welcome to try it!
- [2025.06] Paper is accepted by ICCV 2025!

---

## 🚀 Getting Started

* [Installation](docs/installation.md)
* [Download dataset](docs/dataset.md)
* [Training and evaluation](docs/training_and_evaluation.md)
* To deploy using Gradio, run the following command:

    ```bash
    python app.py --version='./exps/AffordanceVLM-7B'
    ```



## 📊 Main Results
### 🔹 Affordance Segmentation
| Method                               | HANDAL gIoU | HANDAL cIoU | HANDAL† gIoU | HANDAL† cIoU | GraspNet seen gIoU | GraspNet seen cIoU | GraspNet novel gIoU | GraspNet novel cIoU | 3DOI gIoU | 3DOI cIoU |
|--------------------------------------|-------------|-------------|---------------|---------------|----------------------|----------------------|------------------------|------------------------|------------|------------|
| AffordanceNet | 60.3| 60.8 |60.5|60.3|63.3 |64.0| 45.6 |33.2  | 37.4| 37.4 |

### 🔸 Reasoning-Based Affordance Segmentation

| Method  | HANDAL (easy) gIoU | HANDAL (easy) cIoU | HANDAL (hard) gIoU | HANDAL (hard) cIoU | 3DOI gIoU | 3DOI cIoU |
|---------|---------------------|---------------------|---------------------|---------------------|-----------|-----------|
| AffordanceNet| 58.3| 58.1 | 58.2| 57.8 | 38.1 | 39.4|


## 📚 Citation
If you find our work useful, please consider citing:

```bibtex
@inproceedings{wu2025ragnet,
  title={RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping},
  author={Wu, Dongming and Fu, Yanping and Huang, Saike and Liu, Yingfei and Jia, Fan and Liu, Nian and Dai, Feng and Wang, Tiancai and Anwer, Rao Muhammad and Khan, Fahad Shahbaz and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11980--11990},
  year={2025}
}
```

## 🙏 Acknowledgements
We thank the authors that open the following projects. 
- [LISA](https://github.com/dvlab-research/LISA)
- [LLaVA](https://github.com/haotian-liu/LLaVA) 
- [SAM](https://github.com/facebookresearch/segment-anything)