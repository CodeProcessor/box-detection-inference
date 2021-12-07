from unittest import TestCase
from box_detection import BoxDetection
from PIL import Image, ImageDraw
import numpy as np
import cv2


def find_iou(mask1, mask2):
    mask1, mask2 = mask1[:, :, 0], mask2[:, :, 0]
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou


def mask_to_bbox(mask_path):
    mask = np.asarray(Image.open(mask_path))
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray*200, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    xmin, ymin, xmax, ymax = x, y, x+w, y+h
    mask_box = Image.fromarray(np.zeros(mask.shape, dtype=mask.dtype))
    draw = ImageDraw.Draw(mask_box)
    draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=(1, 1, 100))
    mask_box.save("expected_" + mask_path, "JPEG")
    return np.asarray(mask_box)


class TestBoxDetection(TestCase):
    def test_get_boxes(self):
        img_path = '1.jpg'
        annotation_path = '1.png'
        model_file_path = 'best_640.torchscript.pt'
        img = Image.open(img_path)
        bbox_expected = mask_to_bbox(annotation_path)

        box_det = BoxDetection(model_file_path)
        xmin, ymin, xmax, ymax = box_det.get_boxes(img)["points"][0]
        mask = Image.fromarray(np.zeros(bbox_expected.shape, dtype=bbox_expected.dtype))
        draw = ImageDraw.Draw(mask)
        draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=(1, 100, 1))
        bbox_pred = np.asarray(mask)
        mask.save("pred_" + annotation_path, "JPEG")
        iou = find_iou(bbox_pred, bbox_expected)
        print('iou:', iou)
        self.assertGreaterEqual(iou, 0.8)
