from .YOLOv3 import YOLOv3
from .YOLOv3_ov import YOLOv3ov


__all__ = ['build_detector', 'build_detector_ov']

def build_detector(cfg, use_cuda):
    return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES,
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, 
                    is_xywh=True, use_cuda=use_cuda)

def build_detector_ov():
    return YOLOv3ov()
