# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlserver. If not, see <https://www.gnu.org/licenses/>.


# External imports
import numpy as np


def imagenet_preprocess(img: np.array, frame_assets: dict):
    """
    img: np.array
        The input image in (H, W, C)

    Returns: np.array
        The input tensor in (1, C, H, W)
    """
    img = img / 255.0  # in [0, 1], as float32

    # Apply the ImageNet normalization
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    img = (img - imagenet_mean) / imagenet_std

    # Pad the image to a square image
    largest_size = max(img.shape[0], img.shape[1])
    pad_w = largest_size - img.shape[0]
    pad_h = largest_size - img.shape[1]
    img = np.pad(
        img,
        pad_width=(
            (pad_w // 2, pad_w - pad_w // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (0, 0),
        ),
    )

    if largest_size > 224:
        # If the image is larger than 224, we center crop

        # Crop centered window 224x224
        # Borrowed from https://github.com/onnx/models
        # which is Licensed Apache 2
        def crop_center(image, crop_w, crop_h):
            h, w, c = image.shape
            start_y = h // 2 - crop_h // 2
            start_x = w // 2 - crop_w // 2
            return image[start_y : start_y + crop_h, start_x : start_x + crop_w, :]

        img = crop_center(img, 224, 224)
    else:
        # Otherwise we pad again
        # Pad, if necessary, the image to be (224, 224)
        pad_w = 224 - img.shape[0]
        pad_h = 224 - img.shape[1]
        img = np.pad(
            img,
            pad_width=(
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (0, 0),
            ),
        )

    # Finally convert the image from (H, W, C) to (1, C, H, W)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, ...]
    img = img.astype("float32")

    return img


def load_function(preprocessing_name: str):
    if preprocessing_name == "None":
        return lambda x: x
    else:
        return eval(preprocessing_name)
