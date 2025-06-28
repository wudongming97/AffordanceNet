export PATH=/data/cuda/cuda-11.7/cuda/bin:$PATH
export LD_LIBRARY_PATH=/data/cuda/cuda-11.7/cuda/lib64:$LD_LIBRARY_PATH

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=23996 train_aff.py \
--version="./LLaVA/LLaVA-Lightning-7B-v1-1/" \
--vision_pretrained="ckpts/sam_vit_h_4b8939.pth" \
--dataset_dir='./data' \
--dataset="sem_seg||refer_seg||vqa||reason_seg||aff_seg||reason_aff" \
--sample_rates="3,1,1,1,9,3" \
--aff_seg_data="handal||openx||egoobjects||graspnet||rlbench" \
--aff_sample_rates='2,2,4,2,1' \
--reason_aff_data="handal_hard_reasoning||egoobjects_easy_reasoning||egoobjects_hard_reasoning" \
--reason_aff_sample_rates='1,1,1' \
--exp_name="AffordanceVLM-7B" \
--batch_size=40 \
--grad_accumulation_steps=1 \
--epochs=10