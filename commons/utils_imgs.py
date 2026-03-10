#!/usr/bin/env python3
import base64
from io import BytesIO

# import cv2
from PIL import Image
import numpy as np

def encode_image(image_path: str) -> str:
    """画像をbase64エンコード"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def crop_image(image_path: str, box_info: list) -> str:
    """画像をクロップしてbase64エンコード"""
    img = Image.open(image_path) # 日本語対策
    img_array = np.array(img)
    _, s_x, s_y, e_x, e_y = box_info
    s_x, s_y, e_x, e_y = int(s_x * img_array.shape[1] * 0.9 * 0.001), int(s_y * img_array.shape[0] * 0.9 * 0.001), int(e_x * img_array.shape[1] * 1.1 * 0.001), int(e_y * img_array.shape[0] * 1.1 * 0.001)
    cropped = img_array[s_y:e_y, s_x:e_x]
    cropped_img = Image.fromarray(cropped)
    buffered = BytesIO()
    cropped_img.save(buffered, format="PNG")

    # cropped_img.save(f"./test_output/imgs/{os.path.basename(image_path)}")  # for test
    cropped_img.show()  # for test
    # cropped_img.close()

    return base64.b64encode(buffered.getvalue()).decode("utf-8")