## Training and Evaluation

### Pre-trained Weights
#### LLaVA
For convenience of using pre-trained LLaVA weights, we provide a link from [Hugging Face](https://huggingface.co/Dongming97/LLaVA-Lightning-7B-v1-1).

#### SAM
Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).


### Training
To train AffordanceVLM, you can use the following command.
```
bash ./scripts/train.sh
```
When training is finished, to get the full model weight:

```
cd ./runs/AffordanceVLM-7B/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```

For example:
```
CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py  \
    --version="./LLaVA/LLaVA-Lightning-7B-v1-1" \
    --weight="./runs/AffordanceVLM-7B/pytorch_model.bin"  \
    --save_path="./exps/AffordanceVLM-7B"
```

### Evaluation
To evaluate AffordanceVLM on the entire [HANDAL](https://github.com/NVlabs/HANDAL) dataset, please adjust the `--dataset_dir` parameter in `evaluate.sh`.
```
bash ./scripts/evaluate.sh
```

To chat with [AffordanceVLM-7B](https://huggingface.co/Dongming97/AffordanceVLM):
```
CUDA_VISIBLE_DEVICES=0 python chat.py --version=./exps/AffordanceVLM-7B
```

### Main Results

HANDAL:

|      Method      | gIoU | cIoU |
|:----------------:|:----:|-----:|
| AffordanceVLM-7B | 60.3 | 60.8 |   