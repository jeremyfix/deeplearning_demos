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
import transformers
import torch


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


class Transformers:
    def __init__(self, modelname: str, max_new_tokens: int, **kwargs):
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(**kwargs)

        self.model = self.model.to(self.device)

    def __call__(self, inp_data, frame_assets: dict):
        with torch.no_grad():
            # inp_data = inp_data.to(self.device)
            out_model = self.model.generate(
                inp_data, max_new_tokens=self.max_new_tokens
            )
            out_model = out_model.to("cpu")
            frame_assets["outputs"] = out_model


def load_model(cls: str, modelname: str, params: dict):
    if params:
        return eval(f"{cls}(modelname, **params)")
    else:
        return eval(f"{cls}(modelname)")


def test_mobilenet():
    import matplotlib.pyplot as plt
    from PIL import Image
    from . import preprocessing
    from . import postprocessing

    url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    input_field_name = "data"
    model = ONNX("mobilnetv2.7", url, input_field_name)

    assets = {}
    fn_preprocessing = preprocessing.load_function("imagenet_preprocess")

    # image_str = "people.jpeg"
    image_str = "mamba.jpg"
    X = Image.open(image_str)  # .resize((640, 640))
    X = np.array(X)
    X = fn_preprocessing(X, assets)

    model(X, assets)

    label_url = "https://raw.githubusercontent.com/onnx/models/main/vision/classification/synset.txt"

    fn_postprocessing = postprocessing.load_function(
        "label_on_image", {"labels_from_url": label_url}
    )
    out = fn_postprocessing(assets)

    plt.figure()
    plt.imshow(out)
    plt.show()


def test_yolov8():
    # Note :
    # see
    # https://dev.to/andreygermanov/how-to-implement-instance-segmentation-using-yolov8-neural-network-3if9
    import matplotlib.pyplot as plt
    from PIL import Image
    from . import preprocessing
    from . import postprocessing

    url = "https://github.com/jeremyfix/onnx_models/raw/main/Vision/Segmentation/Yolov8/yolov8n-seg.onnx"
    input_field_name = "images"
    model = ONNX("yolov8n-seg", url, input_field_name)

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
        "yolov8_seg", {"labels_from_url": label_url}
    )

    img = fn_postprocessing(assets)

    plt.figure()
    plt.imshow(img)
    plt.show()


def test_yolov11():
    # Note :
    # see
    import matplotlib.pyplot as plt
    from PIL import Image
    from . import preprocessing
    from . import postprocessing

    url = "https://github.com/jeremyfix/onnx_models/raw/main/Vision/ObjectDetection/Yolov11/yolo11n-obb.onnx"
    input_field_name = "images"
    model = ONNX("yolo11n-obb", url, input_field_name)

    assets = {}
    preprocessing_params = [
        {"square_pad": {}},
        {"resize": {"width": 1024, "height": 1024}},
        {"save_asset": {"key": "resized_img"}},
        {"scale": {"value": 255.0}},
        {"transpose": {"dims": [2, 0, 1]}},
        {"add_frontdim": {}},
        {"astype": {"ttype": "float32"}},
    ]
    fn_preprocessing = preprocessing.load_function(preprocessing_params)

    # image_str = "people.jpeg"
    image_str = "bus.jpg"
    X = Image.open(image_str)  # .resize((640, 640))
    X = np.array(X)
    X = fn_preprocessing(X, assets)

    model(X, assets)

    label_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml"
    fn_postprocessing = postprocessing.load_function(
        "yolo11_obbox", {"labels_from_url": label_url}
    )

    img = fn_postprocessing(assets)

    plt.figure()
    plt.imshow(img)
    plt.show()


def test_translation():
    from . import preprocessing, postprocessing

    model_params = {"pretrained_model_name_or_path": "t5-base", "max_new_tokens": 2048}
    model = load_model("Transformers", "translate", model_params)

    preprocessing_params = [
        {"preprompt": {"preprompt": "translate English to French: "}},
        {
            "tokenize": {
                "pretrained_model_name_or_path": "t5-base",
                "model_max_length": 1024,
            }
        },
    ]
    fn_preprocessing = preprocessing.load_function(preprocessing_params)

    postprocessing_params = {
        "pretrained_model_name_or_path": "t5-base",
        "model_max_length": 1024,
        "skip_special_tokens": True,
    }
    fn_postprocessing = postprocessing.load_function("decode", postprocessing_params)
    assets = {}
    inp_data = "This is really a nice time for AI. Language models are impressive in their ability to solve difficult natural language tasks."

    out_pre = fn_preprocessing(inp_data, assets)
    model(out_pre, assets)
    out_post = fn_postprocessing(assets)

    print(f"Input : \n{inp_data}")
    print(f"Output : \n{out_post}")


if __name__ == "__main__":
    # test_mobilenet()
    # test_yolov8()
    # test_translation()
    test_yolov11()
