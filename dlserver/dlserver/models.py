# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlserver. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import urllib.request as request
import tempfile
import logging
import pathlib

# External imports
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
        self.session = ort.InferenceSession(
            str(filepath), providers=ort.get_available_providers()
        )
        self.input_field_name = input_field_name

    def __call__(self, inp_data, frame_assets: dict):
        outputs = self.session.run(None, {self.input_field_name: inp_data})
        frame_assets["output"] = outputs[0]


def load_model(cls: str, modelname: str, params: dict):
    if params:
        return eval(f"{cls}(modelname, **params)")
    else:
        return eval(f"{cls}(modelname)")
