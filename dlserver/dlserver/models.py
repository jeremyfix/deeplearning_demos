# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlserver. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import urllib.request as request


class ImageToLabel:
    def __init__(self):
        pass

    def __call__(self, inp_data, frame_assets: dict):
        frame_assets["label"] = "Example text"


class ONNX:
    def __init__(self, url, labels_from_url=None, labels=None):
        if labels_from_url is not None:
            raw_labels = request.urlopen(labels_from_url).readlines()
            self.labels = [
                " ".join(l.decode("ascii").rstrip().split(" ")[1:]) for l in raw_labels
            ]
        else:
            self.labels = None

    def __call__(self, inp_data, frame_assets: dict):
        # TODO: get cls_id as the argmax of the output of the model
        cls_id = 0
        frame_assets["label"] = self.labels[cls_id]


def load_model(cls: str, params: dict):
    if params:
        return eval(f"{cls}(**params)")
    else:
        return eval(f"{cls}()")
