# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlserver. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import urllib.request as request

# External imports
import cv2


class label_on_image:
    def __init__(self, labels_from_url: str):
        raw_labels = request.urlopen(labels_from_url).readlines()
        self.labels = [
            " ".join(l.decode("ascii").rstrip().split(" ")[1:]) for l in raw_labels
        ]

    def __call__(self, frame_assets: dict):
        """
        frame_assets: dict
            Should have the keys :
                src_img : nd array (H, W, C)
                label: str
        """

        src_img = frame_assets["src_img"]

        scores = frame_assets["output"]
        cls_id = scores.argmax()
        label = self.labels[cls_id]

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, src_img.shape[0] - 10)
        fontScale = 1
        fontColor = (255, 0, 255)
        thickness = 3
        lineType = 2
        result = cv2.putText(
            src_img,
            label,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        return result


def load_function(postprocessing_name: str, params: dict):
    if postprocessing_name == "None":
        return lambda x: x
    else:
        return eval(f"{postprocessing_name}(**params)")
