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
import yaml
import transformers

# Local import
from . import preprocessing


def iou(box1, box2) -> float:
    """
    x1, y1 : bottom left corner
    x2, y2 : top right corner
    box1: ((x1, y1), (x2, y2))
    box2: ((x1, y1), (x2, y2))

    Where  x1 <= x2
           y1 <= y2
    """
    # print(box1, box2)
    (b1_x1, b1_y1), (b1_x2, b1_y2) = box1
    (b2_x1, b2_y1), (b2_x2, b2_y2) = box2
    max_x1 = max(b1_x1, b2_x1)
    max_y1 = max(b1_y1, b2_y1)
    min_x2 = min(b1_x2, b2_x2)
    min_y2 = min(b1_y2, b2_y2)

    intersection = max(0, (min_x2 - max_x1) * (min_y2 - max_y1))
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area + b2_area - intersection
    return intersection / union


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

        # Computes the probabilties
        scores = scores - scores.max()
        exp_scores = np.exp(scores).squeeze()
        cls_proba = exp_scores / np.sum(exp_scores)

        cls_id = scores.argmax()
        label = f"{self.labels[cls_id]} ({cls_proba[cls_id]:.2})"

        # 237, 201, 72
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, src_img.shape[0] - 10)
        fontScale = 1
        fontColor = (78, 121, 167)
        thickness = 3
        lineType = 2
        bgcolor = (237, 201, 72)
        result = cv2.rectangle(
            src_img,
            (2, src_img.shape[0] - 2),
            (int(0.75 * src_img.shape[1]), src_img.shape[0] - 50),
            bgcolor,
            -1,
        )

        result = cv2.putText(
            result,
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

        if output.dtype == np.int16:
            raw_labels = output
        elif output.dtype == np.float32:
            # get classification labels
            raw_labels = np.argmax(output, axis=0).astype(np.uint8)
        else:
            raise RuntimeError(
                f"Unable to handle outputs of dtype {output.dtype} for segmentation overlay"
            )

        # comput confidence score
        # confidence = float(np.max(output, axis=0).mean())

        # generate segmented image
        if self.colorized:
            result_img = self.palette[raw_labels, :]

            if self.blended:
                result_img = 0.5 * frame_assets["resized_img"] + 0.5 * result_img
                result_img = result_img.astype(np.uint8)

        return result_img


class bbox_overlay:
    def __init__(self):
        pass

    def __call__(self, frame_assets: dict):
        return frame_assets["src_img"]


class yolov8_bbox:
    def __init__(self, labels_from_url: str):
        raw_labels = request.urlopen(labels_from_url)
        yml_content = yaml.safe_load(raw_labels)
        self.labels = [v for k, v in yml_content["names"].items()]

        self.num_classes = len(self.labels)
        self.colors = [
            [np.random.randint(0, 255) for _ in range(3)]
            for _ in range(self.num_classes + 1)
        ]
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.8
        self.height_text_box = 10
        self.width_text_box = 100

    def __call__(self, frame_assets: dict):
        # The first output is 1, 116, 8400
        # For 8400 predicted bounding boxes, each with 116 attributes
        # 4 coordinates: xc, yc, width, height
        # 80 classes confidences

        flattened_output = frame_assets["outputs"][0].squeeze().transpose()  # 8440, 84
        # num_predictions = flattened_output.shape[0]

        boxes = flattened_output[:, : (self.num_classes + 4)]  # 8400, 84
        boxes_coordinates = boxes[:, :4]  # 8400, 4
        boxes_confidences = boxes[:, 4:]  # 8400, 80, probabilities

        # Produce the mask of labels

        # Collect all the masks that have a sufficient confidence
        prob_cls_boxes = []
        for coordi, ci in zip(boxes_coordinates, boxes_confidences):
            best_cls = ci.argmax()
            xc, yc, w, h = coordi
            x1, y1 = int(xc - w // 2), int(yc - h // 2)
            x2, y2 = int(xc + w // 2), int(yc + h // 2)

            pimax = ci[best_cls]
            if pimax >= self.confidence_threshold:
                box = ((x1, y1), (x2, y2))

                # +1 to let the class 0 as the background
                prob_cls_boxes.append((pimax, best_cls + 1, box))

        # Sort the boxes by decreasing probability
        # So that the latter filtering on IOU keep the boxes
        # with the highest probabilities
        prob_cls_boxes.sort(key=lambda pkbm: pkbm[0], reverse=True)

        # Apply NMS
        filtered_prob_cls_boxes = []

        while len(prob_cls_boxes) > 0:
            pkbi = prob_cls_boxes[0]
            filtered_prob_cls_boxes.append(pkbi)
            # Remove all the boxes with IOU >= 0.5
            prob_cls_boxes_new = [
                pkbj
                for pkbj in prob_cls_boxes[1:]
                if iou(pkbi[2], pkbj[2]) < self.iou_threshold
            ]
            prob_cls_boxes = prob_cls_boxes_new

        img = frame_assets["resized_img"]

        # Draw the boxes
        for p, k, b in filtered_prob_cls_boxes:
            (x1, y1), (x2, y2) = b

            color = self.colors[k]
            cv2.rectangle(
                img,
                (int(x1 + self.width_text_box), int(y1 + self.height_text_box)),
                (x1, y1),
                color=color,
                thickness=-1,
            )
            cv2.putText(
                img,
                f"{self.labels[k - 1]} {p:.2f}",
                (x1, y1 + self.height_text_box),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
            )
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=2,
            )
        return img


class yolov8_seg:
    def __init__(self, labels_from_url: str):
        raw_labels = request.urlopen(labels_from_url)
        yml_content = yaml.safe_load(raw_labels)
        self.labels = [v for k, v in yml_content["names"].items()]

        self.num_classes = len(self.labels)
        self.colors = [
            [np.random.randint(0, 255) for _ in range(3)]
            for _ in range(self.num_classes + 1)
        ]
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.8
        self.height_text_box = 10
        self.width_text_box = 100

    def __call__(self, frame_assets: dict):
        # The first output is 1, 116, 8400
        # For 8400 predicted bounding boxes, each with 116 attributes
        # 4 coordinates: xc, yc, width, height
        # 80 classes confidences
        # 32 masks weights

        # The second output is 1, 32, 160, 160 and contain the 32 prototype masks
        # each mask being of size 160 x 160

        flattened_output = frame_assets["outputs"][0].squeeze().transpose()  # 8440, 116
        num_predictions = flattened_output.shape[0]
        num_masks = flattened_output.shape[1] - (4 + self.num_classes)

        boxes = flattened_output[:, : (self.num_classes + 4)]  # 8400, 84
        boxes_coordinates = boxes[:, :4]  # 8400, 4
        boxes_confidences = boxes[:, 4:]  # 8400, 80, probabilities

        masks_weights = flattened_output[:, (self.num_classes + 4) :]  # 8400, 32

        prototype_masks = frame_assets["outputs"][1].squeeze()  # 32, 160, 160
        mask_height, mask_width = prototype_masks.shape[1:]

        weighted_masks = masks_weights @ (
            prototype_masks.reshape(num_masks, -1)
        )  # 8400, 160 x 160
        weighted_masks = 1.0 / (1.0 + np.exp(-weighted_masks))
        weighted_masks = weighted_masks.reshape(
            (num_predictions, mask_height, mask_width)
        )

        # Produce the mask of labels

        # Collect all the masks that have a sufficient confidence
        prob_cls_masks = []
        for coordi, ci, xi in zip(boxes_coordinates, boxes_confidences, weighted_masks):
            best_cls = ci.argmax()
            xc, yc, w, h = coordi
            x1, y1 = int(xc - w // 2), int(yc - h // 2)
            x2, y2 = int(xc + w // 2), int(yc + h // 2)

            # print(coordi)
            pimax = ci[best_cls]
            if pimax >= self.confidence_threshold:
                box = ((x1, y1), (x2, y2))

                binary_mask = np.zeros((640, 640), dtype=bool)
                resized_mask = preprocessing.resize(
                    (xi.squeeze() > 0.5), 640, 640, frame_assets
                )
                binary_mask[y1:y2, x1:x2] = resized_mask[y1:y2, x1:x2]
                # +1 to let the class 0 as the background
                prob_cls_masks.append((pimax, best_cls + 1, box, binary_mask))

        # Sort the masks by decreasing probability
        # So that the latter filtering on IOU keep the boxes
        # with the highest probabilities
        prob_cls_masks.sort(key=lambda pkbm: pkbm[0], reverse=True)

        # Apply NMS
        filtered_prob_cls_masks = []

        while len(prob_cls_masks) > 0:
            pkbmi = prob_cls_masks[0]
            filtered_prob_cls_masks.append(pkbmi)
            # Remove all the boxes with IOU >= 0.5
            prob_cls_masks_new = [
                pkbmj
                for pkbmj in prob_cls_masks[1:]
                if iou(pkbmi[2], pkbmj[2]) < self.iou_threshold
            ]
            prob_cls_masks = prob_cls_masks_new
        # print(f"Kept : {len(filtered_prob_cls_masks)} bounding boxes")

        img = frame_assets["resized_img"]
        img = (img * 0.5).astype(np.uint8)

        # Draw the boxes
        for p, k, b, m in filtered_prob_cls_masks:
            (x1, y1), (x2, y2) = b

            color = self.colors[k]
            img[m] = 0.5 * img[m] + 0.5 * np.array(color)

            cv2.rectangle(
                img,
                (int(x1 + self.width_text_box), int(y1 + self.height_text_box)),
                (x1, y1),
                color=color,
                thickness=-1,
            )
            cv2.putText(
                img,
                f"{self.labels[k - 1]} {p:.2f}",
                (x1, y1 + self.height_text_box),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
            )
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=2,
            )
        return img


class yolo11_obbox:
    def __init__(self, labels_from_url: str):
        raw_labels = request.urlopen(labels_from_url)
        yml_content = yaml.safe_load(raw_labels)
        self.labels = [v for k, v in yml_content["names"].items()]

        self.num_classes = len(self.labels)
        self.colors = [
            [np.random.randint(0, 255) for _ in range(3)]
            for _ in range(self.num_classes + 1)
        ]
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.8
        self.height_text_box = 10
        self.width_text_box = 100

    def __call__(self, frame_assets: dict):
        # The first output is 1, 20, 21504
        # For 21504 predicted bounding boxes, each with 20 attributes
        # ?? coordinates
        # ?? classes confidences
        print(frame_assets["outputs"][0].shape)
        # print(frame_assets)
        flattened_output = frame_assets["outputs"][0].squeeze().transpose()  # 8440, 84
        num_predictions = flattened_output.shape[0]
        print(num_predictions)

        img = frame_assets["resized_img"]
        return img


class decode:
    def __init__(self, **kwargs):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(**kwargs)

    def __call__(self, frame_assets: dict):
        return self.tokenizer.decode(frame_assets["outputs"][0])


def load_function(postprocessing_name: str, params: dict):
    if postprocessing_name == "None":
        return lambda x: x
    else:
        if params is None:
            return eval(f"{postprocessing_name}()")
        else:
            return eval(f"{postprocessing_name}(**params)")
