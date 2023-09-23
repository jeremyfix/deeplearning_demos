# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlserver. If not, see <https://www.gnu.org/licenses/>.

# External imports
import cv2


def label_on_image(frame_assets: dict):
    """
    frame_assets: dict
        Should have the keys :
            src_img : nd array (H, W, C)
            label: str
    """

    src_img = frame_assets["src_img"]
    label = frame_assets["label"]

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


def load_function(postprocessing_name: str):
    if postprocessing_name == "None":
        return lambda x: x
    else:
        return eval(postprocessing_name)
