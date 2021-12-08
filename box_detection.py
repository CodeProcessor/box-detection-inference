#!/usr/bin/env python3
"""
@Filename:    box_detection.py
@Author:      dulanj
@Time:        07/12/2021 23:43
"""

import torch
from torchvision import transforms

from utils import non_max_suppression


class BoxDetection:
    model = None
    classes = {
        "0": "box"
    }

    def __init__(self, model_file_path):

        if BoxDetection.model is None:
            BoxDetection.model = torch.jit.load(model_file_path)

        self.model = self.model.eval()

    def get_boxes(self, img, threshold=0.8) -> list:
        """ Runs Yolo Algorithms on PDF images
        Returns:
            Dictionary : {bbox_coordinates, confidence_score,image_original_size}
         """

        ori_w, ori_h = img.size

        # Trained Yolo 640--> Network Input 640
        w_gain = ori_w / 640
        h_gain = ori_h / 640

        preproces_image = transforms.Compose(

            [
                transforms.Resize((640, 640)),
                transforms.ToTensor()
            ]
        )

        input_tensor = preproces_image(img)

        # adding batch dimensions
        input_tensor = input_tensor.unsqueeze(0)

        predictions = self.model(input_tensor)

        prediction_array = non_max_suppression(predictions[0], conf_thres=0.40)[0].cpu().detach().numpy()
        pass
        results = []
        for obj in prediction_array:
            if obj[4] >= threshold:
                xtl = int(max(obj[0], 0) * w_gain)
                ytl = int(max(obj[1], 0) * h_gain)
                xbr = int(min(obj[2], ori_w) * w_gain)
                ybr = int(min(obj[3], ori_h) * h_gain)
                obj_class = str(int(obj[5]))

                results.append({
                    "confidence": str(obj[4]),
                    "label": BoxDetection.classes[obj_class],
                    "points": [xtl, ytl, xbr, ybr],
                    "type": "rectangle",
                })
        return results

        # results = []
        # for obj in objects:
        #     if obj['confidence'] >= threshold:
        #         xtl = max(obj['xmin'], 0)
        #         ytl = max(obj['ymin'], 0)
        #         xbr = min(obj['xmax'], image.width)
        #         ybr = min(obj['ymax'], image.height)
        #         obj_class = int(obj['class_id'])
        #
        #         results.append({
        #             "confidence": str(obj['confidence']),
        #             "label": self.labels.get(obj_class, "unknown"),
        #             "points": [xtl, ytl, xbr, ybr],
        #             "type": "rectangle",
        #         })
        #
        # return results
        # print(prediction_array)
        #
        # bbox_coord = []
        # conf_score = []
        #
        # for array in prediction_array:
        #     array[0] *= w_gain
        #     array[2] *= w_gain
        #
        #     array[1] *= h_gain
        #     array[3] *= h_gain
        #     bbox_coord.append(array[:4])
        #     conf_score.append(array[4])
        #
        # _box_data = {"points": bbox_coord,
        #              "score": conf_score,
        #              "shape": (ori_w, ori_h)
        #              }
        #
        # return _box_data
