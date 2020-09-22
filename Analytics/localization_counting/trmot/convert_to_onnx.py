
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm

from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.utils import *
from utils.io import read_results
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
import torch
from track import eval_seq

import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm

import torch
from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
from utils.utils import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='demo.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/jde_864x480_uncertainty.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--input-video', type=str, help='path to the input video')
    parser.add_argument('--output-format', type=str, default='text', choices=['video', 'text'], help='Expected output format. Video or text.')
    parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
    parser.add_argument("--count", action="store_true", default=False, help='Enable the counting output')
    opt = parser.parse_args()


    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    tracker = JDETracker(opt)

    tracker.model.eval()
    device = 'cuda'
    # ------------------------ export -----------------------------
    output_onnx = 'trmot.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ['images']
    output_names = ['scores']
    #inputs = torch.randn(1, 3, detector.net.width, detector.net.height).to(device)
    inputs = torch.randn(1, 3, 480, 864).to(device)



    torch_out = torch.onnx._export(tracker.model, inputs, output_onnx, export_params=True, verbose=False, opset_version=11, enable_onnx_checker=False,
                                   input_names=input_names, output_names=output_names)
    