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
    from PIL import Image

    url = "https://github.com/jeremyfix/onnx_models/raw/main/Vision/Segmentation/Yolov8/yolov8n-seg.onnx"
    input_field_name = "images"
    model = ONNX("yolov8n", url, input_field_name)

    X = Image.open("bus.jpg").resize((640, 640))

    X = np.array(X)
    X = X / 255.0
    X = X.transpose(2, 0, 1)
    X = X[np.newaxis, ...]
    X = X.astype("float32")
    print(X.shape)

    # X = np.random.random((1, 3, 640, 640)).astype("float32")
    assets = {}
    model(X, assets)

    # The first output is 1, 116, 8400
    # For 8400 predicted bounding boxes, each with 116 attributes
    # 4 coordinates: xc, yc, width, height
    # 80 classes confidences
    # 32 masks weights

    # The second output is 1, 32, 160, 160 and contain the 32 prototype masks
    # each mask being of size 160 x 160

    flattened_output = assets["outputs"][0].squeeze()  # 1, 116, 8400
    boxes = flattened_output[:84, :].transpose()  # 8400, 84
    boxes_coordinates = boxes[:, :4]
    boxes_confidences = boxes[:, 4:]
    masks_weights = flattened_output[84:, :].transpose()  # 8400, 32
    num_predictions = masks_weights.shape[0]
    num_masks = masks_weights.shape[1]

    prototype_masks = assets["outputs"][1].squeeze()  # 32, 160, 160
    mask_height, mask_width = prototype_masks.shape[1:]

    weighted_masks = masks_weights @ (
        prototype_masks.reshape(num_masks, -1)
    )  # 8400, 160 x 160
    weighted_masks = weighted_masks.reshape((num_predictions, mask_height, mask_width))
    print(weighted_masks.shape)

    import matplotlib.pyplot as plt

    plt.figure()
    for ci, xi in zip(boxes_confidences, weighted_masks):
        pimax = 1.0 / (1.0 + np.exp(-ci.max()))
        if pimax >= 0.6:
            best_cls = ci.argmax()
            print(f"plot {pimax}; argmax : {ci.argmax()}")
            if best_cls == 0:
                plt.imshow(255 * (xi > 0.5).astype("uint8"))
    # xi_bus = output_masks[0, 0]
    # plt.imshow(xi_bus)
    plt.show()


if __name__ == "__main__":
    # test_mobilenet()
    test_yolov8()
