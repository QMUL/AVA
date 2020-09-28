import os
import cv2
import time
import argparse
import torch
import numpy as np

from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config

from tqdm import tqdm

class VideoTracker(object):
	def __init__(self, cfg, args):
		self.cfg = cfg
		self.args = args

		if self.args.ov:
			from detector import build_detector_ov
			self.detector = build_detector_ov()
			
		else:
			use_cuda = args.use_cuda and torch.cuda.is_available()
			from detector import build_detector
			self.detector = build_detector(cfg, use_cuda=use_cuda)

		if self.args.ov:
			use_cuda = False
		print('CUDA? {}'.format(use_cuda))

		self.deepsort = build_tracker(cfg, use_cuda=use_cuda, ov=self.args.ov)
		#if not use_cuda:
		#    raise UserWarning("Running in cpu mode!")
		
		if args.display:
			cv2.namedWindow("test", cv2.WINDOW_NORMAL)
			cv2.resizeWindow("test", args.display_width, args.display_height)

		self.vdo = cv2.VideoCapture()
		
	   
		#self.class_names = self.detector.class_names


	def __enter__(self):
		#assert os.path.isfile(self.args.video_path), "Error: path error"
		self.vdo.open(self.args.video_path)
		self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

		if self.args.show and self.args.save_path:
			fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
			self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

		assert self.vdo.isOpened()

		if self.args.AVA:
			if self.args.ov:
				if self.args.skip:
					self.outPath = '../../output/deepsort_ov/{}'.format(args.skip)
				else:
					self.outPath = '../../output/deepsort_ov'

			elif not self.args.use_cuda:
				self.outPath = '../../output/deepsort_cpu'

			else:
				if self.args.skip:
					self.outPath = '../../output/deepsort_gpu/{}'.format(args.skip)
				else:
					self.outPath = '../../output/deepsort_gpu'
			
			if self.args.skip==0:
				self.args.skip=1

			if not os.path.exists(self.outPath):
				os.makedirs(self.outPath)
			
			if self.args.video_path!=0:
				self.countFileName = '{}/{}.csv'.format(self.outPath, self.args.video_path.split('/')[-1].split('.mp4')[0])
			else:
				self.countFileName = '{}/results_live_cam.csv'.format(self.outPath)            
			countFile = open(self.countFileName, 'w')
			countFile.close()

		return self

	
	def __exit__(self, exc_type, exc_value, exc_traceback):
		if exc_type:
			print(exc_type, exc_value, exc_traceback)
		

	def run(self):
		#idx_frame = 0
		if self.args.le:
			skip_frames = 10
		num_images = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT)) if args.video_path!=0 else 10000
		ids = set()
		for fr in tqdm(range(num_images)):

			start = time.time()
			if self.vdo.grab():
				ret, ori_im = self.vdo.retrieve()
				if not ret:
					break

				if self.args.le and  fr % skip_frames != 0:
					continue
				
				if self.args.skip!=0 and fr % self.args.skip != 0:
					continue


			im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

			# do detection
			#bbox_xywh (N,4)
			#cls_conf (N,)
			#cls_ids (N,)
			outputs = None
			bbox_xywh, cls_conf, cls_ids = self.detector(im)
		   
			if bbox_xywh is not None:
				
				# select person class
				mask = cls_ids==0

				bbox_xywh = bbox_xywh[mask]
				bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
				cls_conf = cls_conf[mask]

				# do tracking
				outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

				# draw boxes for visualization
				if len(outputs) > 0:
					bbox_xyxy = outputs[:,:4]
					identities = outputs[:,-1]
					#ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
					ids.update(outputs[:,-1])
			
			end = time.time()

			if self.args.AVA:
				self.countFile = open(self.countFileName, 'a')
				txt = '{:.6f}'.format(end-start)
				
				if outputs is not None:
					for box in outputs:
						txt += ',{:.1f},{:.1f},{:.1f},{:.1f},-2,-2,-2,-2,{:d},-2,-2'.format(box[0], box[1], box[2], box[3], box[4])
				
				txt += '\n'
				self.countFile.write(txt)
				self.countFile.close()

			if args.video_path==0:
				if outputs is not None:
					for d in outputs:
						import pdb; pdb.set_trace()
						cv2.rectangle(ori_im, (d[0], d[1]), (d[2], d[3]), (0, 255, 0), 2)
				cv2.imshow('Live', ori_im)
				cv2.waitKey(1)

				   
			if self.args.save_path:
				self.writer.write(ori_im)
			
			if self.args.show:
				cv2.imshow("test", ori_im)
				cv2.waitKey(1)

def parse_args():
	parser = argparse.ArgumentParser()
	#parser.add_argument("VIDEO_PATH", type=str)
	parser.add_argument('--video_path', type=str, required=False, default=0, help='dataset path')
	parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml", help='Configure detector')
	parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml", help='Configure tracker')
	parser.add_argument("--display", action="store_true", default=False, help='Show result')
	parser.add_argument("--display_width", type=int, default=800, help='Display width')
	parser.add_argument("--display_height", type=int, default=600, help='Display height')
	parser.add_argument("--save_path", type=str, help='Saving path for visual result')
	parser.add_argument("--show", action="store_true", default=False, help='Show/save visual results')
	parser.add_argument("--AVA", action="store_true", default=False, help='Enable output for AVA')
	parser.add_argument("--ov", action="store_true", default=False, help='Run in OpenVino')
	parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True, help='Run in CPU')
	parser.add_argument("--le", action="store_true", default=False, help='Low end mode')
	parser.add_argument('--skip', default=0, type=int, help='skip X frames')

	return parser.parse_args()

if __name__=="__main__":
	args = parse_args()
	cfg = get_config()
	cfg.merge_from_file(args.config_detection)
	cfg.merge_from_file(args.config_deepsort)

	with VideoTracker(cfg, args) as vdo_trk:
		vdo_trk.run()
