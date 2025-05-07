#cd ./runs/lisa/ckpt_model
#
#python zero_to_fp32.py . ../pytorch_model.bin
#
#cd ../../..p

CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py  \
--version="./LLaVA/LLaVA-Lightning-7B-v1-1" \
--weight="./runs/AffordanceVLM-7B/pytorch_model.bin"  \
--save_path="./exps/AffordanceVLM-7B"