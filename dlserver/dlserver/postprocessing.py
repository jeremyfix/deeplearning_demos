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

# External imports
import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb


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
                output: nd array (K, )
        """

        src_img = frame_assets["src_img"]

        scores = frame_assets["outputs"][0]
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


def get_palette(num_classes: int):
    # prepare and return palette
    palette = np.zeros((num_classes, 3))

    for hue in range(num_classes):
        if hue == 0:  # Background color
            colors = (0, 0, 0)
        else:
            colors = hsv_to_rgb((hue / num_classes, 0.75, 0.75))

        for i in range(3):
            palette[hue, i] = int(colors[i] * 255)

    return palette


class segmentation_overlay:
    def __init__(self, num_classes: int, colorized: bool, blended: bool):
        self.num_classes = num_classes
        self.palette = get_palette(num_classes)
        self.colorized = colorized
        self.blended = blended

    def __call__(self, frame_assets: dict):
        """
        frame_assets: dict
            Should have the keys :
                src_img : nd array (H, W, C)
                output: nd array (1, K, H, W)
        """

        # src_img = frame_assets["src_img"]
        output = frame_assets["outputs"][0].squeeze()

        # get classification labels
        raw_labels = np.argmax(output, axis=0).astype(np.uint8)

        # comput confidence score
        # confidence = float(np.max(output, axis=0).mean())

        # generate segmented image
        if self.colorized:
            result_img = self.palette[raw_labels, :]

            if self.blended:
                result_img = 0.5 * frame_assets["resized_img"] + 0.5 * result_img

        return result_img


class bbox_overlay:
    def __init__(self):
        pass

    def __call__(self, frame_assets: dict):
        return frame_assets["src_img"]


def load_function(postprocessing_name: str, params: dict):
    if postprocessing_name == "None":
        return lambda x: x
    else:
        if params is None:
            return eval(f"{postprocessing_name}()")
        else:
            return eval(f"{postprocessing_name}(**params)")
