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


logger.setLevel(logging.INFO)


def track(opt): 

	if opt.AVA:            
		if opt.ov:
			if opt.skip:
				outPath = '../../output/trmot_ov/{}'.format(opt.skip)
			else:
				outPath = '../../output/trmot_ov'
		elif opt.cpu:
			if opt.skip:
				outPath = '../../output/trmot_cpu/{}'.format(opt.skip)
			else:
				outPath = '../../output/trmot_cpu'
		else:
			if opt.skip:
				outPath = '../../output/trmot_gpu/{}'.format(opt.skip)
			else:
				outPath = '../../output/trmot_gpu'
		if opt.skip==0:
			opt.skip=1
		if not os.path.exists(outPath):
			os.makedirs(outPath)
		
		if opt.video_path!=0:
			countFileName = '{}/{}.csv'.format(outPath, opt.video_path.split('/')[-1].split('.mp4')[0])
		else:
			countFileName = '{}/results_live_cam.csv'.format(outPath)
		countFile = open(countFileName, 'w')
		countFile.close()


	result_root = opt.output_root if opt.output_root!='' else '.'
	mkdir_if_missing(result_root)

	cfg_dict = parse_model_cfg(opt.cfg)
	opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

	# run tracking
	timer = Timer()
	accs = []
	n_frame = 0

	logger.info('Starting tracking...')
	dataloader = datasets.LoadVideo(opt.video_path, opt.img_size)
	result_filename = os.path.join(result_root, 'results.txt')
	frame_rate = dataloader.frame_rate 

	frame_dir = None if opt.output_format=='text' else osp.join(result_root, 'frame')
	try:
		eval_seq(opt, dataloader, 'mot', result_filename, countFileName, save_dir=frame_dir, show_image=False, frame_rate=frame_rate)
	except Exception as e:
		logger.info(e)

	if opt.output_format == 'video':
		output_video_path = osp.join(result_root, 'result.mp4')
		cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(osp.join(result_root, 'frame'), output_video_path)
		os.system(cmd_str)

		
if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog='demo.py')
	parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
	parser.add_argument('--weights', type=str, default='weights/jde_864x480_uncertainty.pt', help='path to weights file')
	parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
	parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
	parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
	parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
	parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
	parser.add_argument('--video_path', type=str, required=False, default=0, help='dataset path')
	parser.add_argument('--output-format', type=str, default='text', choices=['video', 'text'], help='Expected output format. Video or text.')
	parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
	parser.add_argument("--AVA", action="store_true", default=False, help='Enable output for AVA')
	parser.add_argument("--ov", action="store_true", default=False, help='Run in openvino')
	parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
	parser.add_argument("--le", action="store_true", default=False, help='Low end mode')
	parser.add_argument('--skip', default=0, type=int, help='skip X frames')

	opt = parser.parse_args()
	#print(opt, end='\n\n')

  
	track(opt)

