#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Client script to send an image and prompt to a Flask-based vision-language segmentation server.

from __future__ import absolute_import, print_function, division
import requests
import cv2
import base64
import numpy as np

# ---------------------------
# Encode image to base64 string
# ---------------------------
def img2b64(img):
    retval, buffer = cv2.imencode('.bmp', img)  # Encode as BMP
    pic_str = base64.b64encode(buffer).decode()  # Convert to base64 string
    return pic_str

# ---------------------------
# Decode base64 string back to image
# ---------------------------
def b642img(pic_str):
    img_data = base64.b64decode(pic_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np

# ---------------------------
# Send image and prompt to server, receive result and save
# ---------------------------
def post_files():
    path = 'vis_output/my_workspace.JPG'  # Input image path
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to read image at {path}")
        return

    pic_str = img2b64(img)
    data = {
        'img': pic_str,
        'prompt': 'Please segment the affordance map of mug in this image.'
    }

    # Send POST request to Flask server
    r = requests.post('http://localhost:3200/img_mask', json=data)

    if r.status_code == 200:
        print('Success. Received response from server.')
        result = r.json()
        result_b64 = result.get('img', None)

        if result_b64:
            result_img = b642img(result_b64)
            save_path = 'affordance_mask_result.jpg'
            cv2.imwrite(save_path, result_img)
            print(f"Result saved to {save_path}")
        else:
            print("No image returned in the response.")
    else:
        print(f"Request failed with status code {r.status_code}")

# ---------------------------
# Main entry
# ---------------------------
if __name__ == '__main__':
    post_files()

