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
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the affordance map of {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the affordance map of {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is the affordance map of {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is the affordance map of {class_name} in this image? Please output segmentation mask.",
]


class ReasonAffDataset(torch.utils.data.Dataset):
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
        reason_aff_data="handal_hard_reasoning",
        reason_aff_sample_ratio=[1],
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.reason_aff_data = reason_aff_data
        reason_aff_sample_ratio = np.array(reason_aff_sample_ratio)
        self.reason_aff_sample_ratio = reason_aff_sample_ratio / reason_aff_sample_ratio.sum()
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
        self.answer_list = ANSWER_LIST

        reason_aff_datas = reason_aff_data.split("||")
        self.data2list = {}
        self.object_ids = {}
        for ds in reason_aff_datas:
            if ds == "handal_hard_reasoning" or ds == "egoobjects_easy_reasoning" or ds == "egoobjects_hard_reasoning":
                pkl_path = os.path.join(base_image_dir, f'{ds}_val.pkl')
                images = {}
                labels = {}
                questions = {}
                answers = {}
                with open(pkl_path, 'rb') as f:
                    aff_datas = pickle.load(f)
                for aff_data in aff_datas:
                    if aff_data['task_object_class'] not in images:
                        images[aff_data['task_object_class']] = []
                        labels[aff_data['task_object_class']] = []
                        questions[aff_data['task_object_class']] = []
                        answers[aff_data['task_object_class']] = []
                    images[aff_data['task_object_class']].append(aff_data['frame_path'])
                    labels[aff_data['task_object_class']].append(aff_data['mask_path'])
                    questions[aff_data['task_object_class']].append(aff_data['question'])
                    answers[aff_data['task_object_class']].append(aff_data['answer'])
                # keep same numbers of samples for each class
                for k in images.keys():
                    assert len(images[k]) == len(labels[k])
                self.data2list[ds] = (images, labels, questions, answers)
                print(f"categories of {ds}: ", images.keys())
                print(f"number of {ds} samples: ", len(aff_datas))
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
        ds = np.random.choice(list(self.data2list.keys()), p=self.reason_aff_sample_ratio)

        images, labels, my_questions, my_answers = self.data2list[ds]
        class_name = random.choice(list(images.keys()))
        idx = random.randint(0, len(images[class_name]) - 1)
        image_path = images[class_name][idx]
        label_path = labels[class_name][idx]
        my_question = my_questions[class_name][idx]
        my_answer = my_answers[class_name][idx]

        # load image and prepare input for clip and sam
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

            # assert len(text.split("||")) == 1
            # question_template = random.choice(self.affordance_question_list)
            # questions.append(question_template.format(class_name=text.lower()))
            #
            # answers.append(random.choice(self.answer_list))
            questions.append(DEFAULT_IMAGE_TOKEN + "\n" + "You are an embodied robot. " + my_question)
            # answers.append(my_answer + " [SEG].")
            answers.append(my_answer + " [AFF].")

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


class ReasonAffValDataset(torch.utils.data.Dataset):
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

        self.images = []
        self.labels = []
        self.questions = []
        self.answers = []
        self.class_ids = []
        self.class_names = []
        pkl_path = os.path.join(self.base_image_dir, f'{ds}_val.pkl')
        with open(pkl_path, 'rb') as f:
            reason_datas = pickle.load(f)
        for reason_data in reason_datas:
            self.images.append(reason_data['frame_path'])
            self.labels.append(reason_data['mask_path'])
            self.questions.append(reason_data['question'])
            self.answers.append(reason_data['answer'])
            self.class_ids.append(None)
            self.class_names.append(reason_data['task_object_class'])

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

        # load question and answer
        my_question = self.questions[idx]
        my_answer = self.answers[idx]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()

            conv.append_message(
                conv.roles[0],
                DEFAULT_IMAGE_TOKEN + "\n" + "You are an embodied robot. " + "{}".format(my_question),
            )
            conv.append_message(conv.roles[1], my_answer + " [AFF].")
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