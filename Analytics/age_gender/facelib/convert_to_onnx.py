from Retinaface.Retinaface import FaceDetector
from AgeGender.Detector import AgeGender
import cv2
from time import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# Own libraries
from libs.reader import *


# Parse arguments
def get_args():
	parser = argparse.ArgumentParser(description='Age/Gender estimation')
	parser.add_argument('--margin', type=float, default=0, help='Margin around detected face for age-gender estimation')
	parser.add_argument('--img_size', type=int, default=112)
	parser.add_argument('--video_path', type=str, default=None, help='Video path')
	parser.add_argument('--plot', action='store_true', default=False, help='Enable age estimation')
	parser.add_argument('--age', action='store_true', default=False, help='Enable age estimation')
	parser.add_argument('--ov', action='store_true', default=False, help='Run in openvino')
	parser.add_argument('--cpu', action='store_true', default=False, help='Run with CPU')
	return parser.parse_args()


if __name__ == '__main__':

	args = get_args()
	device='cuda'

	age_gender = AgeGender(name='full', weight_path='weights/ShufflenetFull.pth', device=device)

	# ------------------------ export -----------------------------
	output_onnx = 'facelib.onnx'
	print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
	input_names = ['images']
	output_names = ['scores']
	inputs = torch.randn(1, 3, args.img_size, args.img_size).to(device)

	torch_out = torch.onnx._export(age_gender.model, inputs, output_onnx, export_params=True, verbose=False, input_names=input_names, output_names=output_names)
 