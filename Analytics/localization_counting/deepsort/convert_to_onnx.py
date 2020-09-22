import os
import cv2
import time
import argparse
import torch
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config

from tqdm import tqdm



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml", help='Configure detector')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml", help='Configure tracker')
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True, help='For devices without screen')
    #parser.add_argument("--frame_interval", type=int, default=1, help='To jump frames')
    parser.add_argument("--display_width", type=int, default=800, help='Display width')
    parser.add_argument("--display_height", type=int, default=600, help='Display height')
    parser.add_argument("--save_path", type=str, help='Saving path for visual result')
    parser.add_argument("--show", action="store_true", default=False, help='Show/save visual results')
    parser.add_argument("--count", action="store_true", default=False, help='Enable the counting output')
    parser.add_argument("--ov", action="store_true", default=False, help='Run in OpenVino')
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True, help='Run in CPU')
    args = parser.parse_args()


    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    use_cuda = args.use_cuda and torch.cuda.is_available()
    torch.set_grad_enabled(False)
    #model = build_detector(cfg, args.ov, use_cuda=use_cuda)
    model = build_tracker(cfg, use_cuda=use_cuda)
   

    model.extractor.net.eval()
    device = 'cuda'
    # ------------------------ export -----------------------------
    output_onnx = 'deep.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ['images']
    output_names = ['scores']
    #inputs = torch.randn(1, 3, detector.net.width, detector.net.height).to(device)
    inputs = torch.randn(1, 3, 128, 64).to(device)



    torch_out = torch.onnx._export(model.extractor.net, inputs, output_onnx, export_params=True, verbose=False, 
                                   input_names=input_names, output_names=output_names)
    