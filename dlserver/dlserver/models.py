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
import urllib.request as request
import tempfile
import logging
import pathlib

# External imports
import numpy as np
import onnxruntime as ort


class ImageToLabel:
    def __init__(self, modelname: str):
        pass

    def __call__(self, inp_data, frame_assets: dict):
        frame_assets["label"] = "Example text"


class ONNX:
    def __init__(self, modelname: str, url: str, input_field_name: str):
        # Download the onnx model
        filepath = pathlib.Path(tempfile.gettempdir()) / f"{modelname}.onnx"
        print(filepath)
        if not filepath.exists():
            logging.debug(f"Downloading {url} into {filepath}")
            request.urlretrieve(url, filename=filepath)
        logging.debug(f"Available ORT providers : {ort.get_available_providers()}")
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider"]
        elif "CPUExecutionProvider":
            providers = ["CPUExecutionProvider"]
        else:
            providers = available_providers
        logging.debug(f"I will be using the ORT providers : {providers}")
        self.session = ort.InferenceSession(str(filepath), providers=providers)
        self.input_field_name = input_field_name

    def __call__(self, inp_data, frame_assets: dict):
        outputs = self.session.run(None, {self.input_field_name: inp_data})
        frame_assets["outputs"] = outputs


def load_model(cls: str, modelname: str, params: dict):
    if params:
        return eval(f"{cls}(modelname, **params)")
    else:
        return eval(f"{cls}(modelname)")


def test_mobilenet():
    url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    input_field_name = "data"
    model = ONNX("mobilnetv2.7", url, input_field_name)

    X = np.zeros((1, 3, 224, 224), dtype="float32")
    assets = {}
    model(X, assets)
    for o in assets["outputs"]:
        print(o.shape)


def test_yolov8():
    # Note :
    # see
    # https://dev.to/andreygermanov/how-to-implement-instance-segmentation-using-yolov8-neural-network-3if9
    import matplotlib.pyplot as plt
    from PIL import Image
    from . import preprocessing
    from . import postprocessing

    url = "https://github.com/jeremyfix/onnx_models/raw/main/Vision/ObjectDetection/Yolov8/yolov8n.pt"
    # url = "https://github.com/jeremyfix/onnx_models/raw/main/Vision/Segmentation/Yolov8/yolov8n-seg.onnx"
    input_field_name = "images"
    print(url)
    model = ONNX("yolov8n", url, input_field_name)

    assets = {}
    preprocessing_params = [
        {"square_pad": {}},
        {"resize": {"width": 640, "height": 640}},
        {"save_asset": {"key": "resized_img"}},
        {"scale": {"value": 255.0}},
        {"transpose": {"dims": [2, 0, 1]}},
        {"add_frontdim": {}},
        {"astype": {"ttype": "float32"}},
    ]
    fn_preprocessing = preprocessing.load_function(preprocessing_params)

    # image_str = "people.jpeg"
    image_str = "animals.jpg"
    X = Image.open(image_str)  # .resize((640, 640))
    X = np.array(X)
    X = fn_preprocessing(X, assets)

    model(X, assets)

    label_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml"
    fn_postprocessing = postprocessing.load_function(
        "yolov8_bbox", {"labels_from_url": label_url}
    )

    img = fn_postprocessing(assets)

    plt.figure()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # test_mobilenet()
    test_yolov8()
