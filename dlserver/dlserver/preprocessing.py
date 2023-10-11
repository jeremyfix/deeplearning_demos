# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# dlserver. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import logging

# External imports
import numpy as np
from PIL import Image


def save_asset(inarray: np.array, key: str, frame_assets: dict):
    frame_assets[key] = inarray.copy()
    return inarray


def resize(inarray: np.array, height: int, width: int, frame_assets: dict):
    inpil = Image.fromarray(inarray)
    inpil = inpil.resize((width, height))
    return np.array(inpil)


def scale(inarray: np.array, value: float, frame_assets: dict):
    return inarray / value


def normalize(img: np.array, mus: list, stds: list, frame_assets: dict):
    return (img - np.array(mus)) / np.array(stds)


def square_pad(img: np.array, frame_assets: dict):
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
    return img


def pad(img: np.array, maxsize: int, frame_assets: dict):
    assert img.shape[0] <= maxsize and img.shape[1] <= maxsize
    pad_h = maxsize - img.shape[0]
    pad_w = maxsize - img.shape[1]
    img = np.pad(
        img,
        pad_width=(
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0),
        ),
    )
    return img


def center_crop(img: np.array, cropsize: int, frame_assets: dict):
    assert img.shape[0] >= cropsize and img.shape[1] >= cropsize
    h, w, c = img.shape
    start_y = h // 2 - cropsize // 2
    start_x = w // 2 - cropsize // 2
    return img[start_y : start_y + cropsize, start_x : start_x + cropsize, :]


def pad_or_crop(img: np.array, targetsize: int, frame_assets: dict):
    largest_size = max(img.shape[0], img.shape[1])
    if largest_size > targetsize:
        return center_crop(img, targetsize, frame_assets)
    else:
        return pad(img, targetsize, frame_assets)


def transpose(img: np.array, dims, frame_assets: dict):
    return img.transpose(*dims)


def astype(img: np.array, ttype: str, frame_assets: dict):
    img = img.astype(ttype)
    return img


def add_frontdim(img: np.array, frame_assets: dict):
    return img[np.newaxis, ...]


def imagenet_preprocess():
    preprocessing_params = [
        {"pad_or_crop": {"targetsize": 224}},
        {"save_asset": {"key": "resized_img"}},
        {"scale": {"value": 255.0}},
        {"normalize": {"mus": [0.485, 0.456, 0.406], "stds": [0.229, 0.224, 0.225]}},
        {"transpose": {"dims": [2, 0, 1]}},
        {"astype": {"ttype": "float32"}},
        {"add_frontdim": {}},
    ]
    return load_compose(preprocessing_params)


def compose(inarray, fns: list, frame_assets: dict):
    for fn in fns:
        inarray = fn(inarray, frame_assets)
    return inarray


def load_compose(preprocessings: list):
    # iterate over the list making up the pipeline
    # and compose all the transforms
    fns = []
    for fn_params in preprocessings:
        fn, params = next(iter(fn_params.items()))
        if params is None:
            fn_i = eval(
                f"lambda inarray, frame_assets, params=params:{fn}(inarray,frame_assets=frame_assets)"
            )
        else:
            fn_i = eval(
                f"lambda inarray, frame_assets, params=params:{fn}(inarray,frame_assets=frame_assets,**params)"
            )
        fns.append(fn_i)
    return lambda inarray, frame_assets: compose(inarray, fns, frame_assets)


def load_function(preprocessings: str | list):
    # preprocessing can be either a string or a dictionary
    logging.debug(f"Loading the preprocessing from {preprocessings}")
    if isinstance(preprocessings, str):
        return eval(preprocessings)()
    else:
        return load_compose(preprocessings)


def test_preprocessing():
    # preprocessing_params = [
    #     {"resize": {"height": 256, "width": 256}},
    #     {"save_asset": {"key": "resize_img"}},
    #     {"scale": {"value": 255.0}},
    #     {"normalize": {"mus": [0.485, 0.456, 0.406], "stds": [0.229, 0.224, 0.225]}},
    #     {"pad_or_crop": {"targetsize": 224}},
    #     # {"transpose": {"dims": [2, 0, 1]}},
    #     {"astype": {"ttype": "float32"}},
    # ]
    # fn_preprocessing = load_function(preprocessing_params)
    fn_preprocessing = load_function("imagenet_preprocess")

    frame_assets = {}
    x = np.random.randint(low=0, high=255, size=(68, 128, 3), dtype=np.uint8)
    y = fn_preprocessing(x, frame_assets)
    # y = (y * 255).astype(np.uint8)
    print(y.dtype, y.shape)
    # Image.fromarray(y).show()
    # print(x, y, y.shape)
    # print(frame_assets)


if __name__ == "__main__":
    test_preprocessing()
