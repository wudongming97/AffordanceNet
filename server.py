#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Flask-based service to receive an image and a prompt, perform vision-language model inference, and return a segmentation mask.

from __future__ import absolute_import, print_function, division
from flask import Flask, request, jsonify
import os
import json
import base64
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from PIL import Image
from io import BytesIO

# Custom model and utility imports
from model.AffordanceVLM import AffordanceVLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
)

app = Flask(__name__)

# ---------------------------
# Argument parser for model config
# ---------------------------
def parse_args(args):
    parser = argparse.ArgumentParser(description="AffordanceVLM Flask Service")
    parser.add_argument("--version", default="/data/AffordanceNet/exps/AffordanceVLM-7B")
    parser.add_argument("--vis_save_path", default="./vis_output/ur5_samples", type=str)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)

# ---------------------------
# Model initialization
# ---------------------------
args = parse_args(None)
os.makedirs(args.vis_save_path, exist_ok=True)

# Load tokenizer and add custom tokens
tokenizer = AutoTokenizer.from_pretrained(args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token
args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
args.aff_token_idx = tokenizer("[AFF]", add_special_tokens=False).input_ids[0]

# Set precision
torch_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.half,
    "fp32": torch.float32
}[args.precision]

# Optional quantization configs
kwargs = {"torch_dtype": torch_dtype}
if args.load_in_4bit:
    kwargs.update({
        "torch_dtype": torch.half,
        "load_in_4bit": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["visual_model"],
        ),
    })
elif args.load_in_8bit:
    kwargs.update({
        "torch_dtype": torch.half,
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["visual_model"],
        ),
    })

# Load model
model = AffordanceVLMForCausalLM.from_pretrained(
    args.version,
    vision_tower=args.vision_tower,
    seg_token_idx=args.seg_token_idx,
    aff_token_idx=args.aff_token_idx,
    low_cpu_mem_usage=True,
    **kwargs
)

# Set special tokens
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Initialize vision modules
model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower().to(dtype=torch_dtype)

# Model precision setup
if args.precision == "bf16":
    model = model.bfloat16().cuda()
elif args.precision == "fp16" and not args.load_in_4bit and not args.load_in_8bit:
    model.model.vision_tower = None
    import deepspeed
    model = deepspeed.init_inference(model=model, dtype=torch.half, replace_with_kernel_inject=True).module
    model.model.vision_tower = vision_tower.half().cuda()
else:
    model = model.float().cuda()

vision_tower.to(device=args.local_rank)
clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
transform = ResizeLongestSide(args.image_size)

model.eval()

# ---------------------------
# Image preprocessing function
# ---------------------------
def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
               pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
               img_size=1024) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    x = F.pad(x, (0, img_size - w, 0, img_size - h))
    return x

# ---------------------------
# Segmentation core logic
# ---------------------------
def segment(image_path, prompt):
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\nYou are an embodied robot. " + prompt
    if args.use_mm_start_end:
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN,
                                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    # CLIP preprocessing
    image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
    image_clip = image_clip.to(dtype=torch_dtype)

    # Resize and normalize
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = preprocess(torch.from_numpy(image).permute(2, 0, 1)).unsqueeze(0).cuda().to(dtype=torch_dtype)

    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()

    # Model inference
    output_ids, pred_masks = model.evaluate(image_clip, image, input_ids, resize_list, original_size_list,
                                            max_new_tokens=512, tokenizer=tokenizer)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False).replace("\n", "").replace("  ", " ")
    print("text_output:", text_output)

    # Save predicted masks
    save_mask_path = ""
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue
        pred_mask = pred_mask.detach().cpu().numpy()[0] > 0
        save_mask_path = f"{args.vis_save_path}/{os.path.basename(image_path).split('.')[0]}_mask_{i}.jpg"
        cv2.imwrite(save_mask_path, pred_mask * 100)
        print(f"Saved: {save_mask_path}")

        save_img_path = f"{args.vis_save_path}/{os.path.basename(image_path).split('.')[0]}_masked_img_{i}.jpg"
        save_img = image_np.copy()
        save_img[pred_mask] = (image_np * 0.5 + pred_mask[:, :, None] * np.array([255, 0, 0]) * 0.5)[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_img_path, save_img)
        print(f"Saved: {save_img_path}")

    return save_mask_path

# ---------------------------
# Convert image to base64
# ---------------------------
def img2b64(img):
    _, buffer = cv2.imencode('.bmp', img)
    return base64.b64encode(buffer).decode()

# ---------------------------
# HTTP endpoint: /img_mask
# ---------------------------
@app.route("/img_mask", methods=['POST', 'GET'])
def recv_json():
    data = json.loads(request.data)
    prompt = data.get('prompt', 'no_recv')
    print("Received prompt:", prompt)

    # Decode base64 image
    img_data = base64.b64decode(data['img'])
    img_np = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(args.vis_save_path, 'img.jpg'), img_np)

    # Run segmentation
    save_path = segment(os.path.join(args.vis_save_path, 'img.jpg'), prompt)
    img = cv2.imread(save_path)
    pic_str = img2b64(img)

    return jsonify({'img': pic_str})

# ---------------------------
# App entry point
# ---------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3200)
