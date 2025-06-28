export PATH=/data/cuda/cuda-11.7/cuda/bin:$PATH
export LD_LIBRARY_PATH=/data/cuda/cuda-11.7/cuda/lib64:$LD_LIBRARY_PATH

affordance_model="./exps/AffordanceVLM-7B"

deepspeed --master_port=24990 train_aff.py \
  --version=$affordance_model \
  --dataset_dir='/mnt/llm3d/lisa_data' \
  --dataset="reason_seg" \
  --sample_rates="1" \
  --vision_pretrained="ckpts/sam_vit_h_4b8939.pth" \
  --exp_name="AffordanceVLM-7B" \
  --eval_only \
  --eval_affordance \
  --val_dataset="handal_all"

#deepspeed --master_port=24991 train_aff.py \
#  --version=$affordance_model \
#  --dataset_dir='/mnt/llm3d/lisa_data' \
#  --dataset="reason_seg" \
#  --sample_rates="1" \
#  --vision_pretrained="ckpts/sam_vit_h_4b8939.pth" \
#  --exp_name="AffordanceVLM-7B" \
#  --eval_only \
#  --eval_affordance \
#  --val_dataset="handal_mini"
#
#deepspeed --master_port=24992 train_aff.py \
#  --version=$affordance_model \
#  --dataset_dir='/mnt/llm3d/lisa_data' \
#  --dataset="reason_seg" \
#  --sample_rates="1" \
#  --vision_pretrained="ckpts/sam_vit_h_4b8939.pth" \
#  --exp_name="AffordanceVLM-7B" \
#  --eval_only \
#  --eval_affordance \
#  --val_dataset="graspnet_test_seen"
#
#deepspeed --master_port=24993 train_aff.py \
#  --version=$affordance_model \
#  --dataset_dir='/mnt/llm3d/lisa_data' \
#  --dataset="reason_seg" \
#  --sample_rates="1" \
#  --vision_pretrained="ckpts/sam_vit_h_4b8939.pth" \
#  --exp_name="AffordanceVLM-7B" \
#  --eval_only \
#  --eval_affordance \
#  --val_dataset="graspnet_test_novel"
#
#deepspeed --master_port=24994 train_aff.py \
#  --version=$affordance_model \
#  --dataset_dir='/mnt/llm3d/lisa_data' \
#  --dataset="reason_seg" \
#  --sample_rates="1" \
#  --vision_pretrained="ckpts/sam_vit_h_4b8939.pth" \
#  --exp_name="AffordanceVLM-7B" \
#  --eval_only \
#  --eval_affordance \
#  --val_dataset="3doi"
#
#deepspeed --master_port=24995 train_aff.py \
#  --version=$affordance_model \
#  --dataset_dir='/mnt/llm3d/lisa_data' \
#  --dataset="reason_seg" \
#  --sample_rates="1" \
#  --vision_pretrained="ckpts/sam_vit_h_4b8939.pth" \
#  --exp_name="AffordanceVLM-7B" \
#  --eval_only \
#  --eval_reason_aff \
#  --val_dataset="handal_hard_reasoning"
#
#deepspeed --master_port=24996 train_aff.py \
#  --version=$affordance_model \
#  --dataset_dir='/mnt/llm3d/lisa_data' \
#  --dataset="reason_seg" \
#  --sample_rates="1" \
#  --vision_pretrained="ckpts/sam_vit_h_4b8939.pth" \
#  --exp_name="AffordanceVLM-7B" \
#  --eval_only \
#  --eval_reason_aff \
#  --val_dataset="handal_easy_reasoning"
#
#deepspeed --master_port=24997 train_aff.py \
#  --version=$affordance_model \
#  --dataset_dir='/mnt/llm3d/lisa_data' \
#  --dataset="reason_seg" \
#  --sample_rates="1" \
#  --vision_pretrained="ckpts/sam_vit_h_4b8939.pth" \
#  --exp_name="AffordanceVLM-7B" \
#  --eval_only \
#  --eval_reason_aff \
#  --val_dataset="3doi_easy_reasoning"
