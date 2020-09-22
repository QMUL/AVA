from models.mtcnn import MTCNN, PNet, RNet, ONet, prewhiten, fixed_image_standardization
from models.utils.detect_face import extract_face
import torch
import numpy as np
import cv2

import os
import argparse
from tqdm import tqdm 
import time


parser = argparse.ArgumentParser(description='FaceNet')
parser.add_argument('--video_path', type=str, required=False, default=0, help='dataset path')
parser.add_argument("--AVA", action="store_true", default=False, help='Enable output for AVA')
parser.add_argument("--ov", action="store_true", default=False, help='Run in openvino')
parser.add_argument("--le", action="store_true", default=False, help='Low end mode')
parser.add_argument("--plot", action="store_true", default=False, help='Plot')
parser.add_argument('--skip', default=0, type=int, help='skip X frames')
parser.add_argument("--cpu", action="store_true", default=False)
args = parser.parse_args()


if args.AVA:
		
	if args.ov or args.cpu:
		if args.ov:
			if args.skip:
				outPath = '../../output/mtcnn_ov/{}'.format(args.skip)
			else:
				outPath = '../../output/mtcnn_ov'
		elif args.skip:
				outPath = '../../output/mtcnn_cpu/{}'.format(args.skip)
		else:
			outPath = '../../output/mtcnn_cpu'
		device = torch.device("cpu")

	else:
		if args.skip:
			outPath = '../../output/mtcnn_gpu/{}'.format(args.skip)
		else:
			outPath = '../../output/mtcnn_gpu'
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if args.skip==0:
		args.skip=1
	if not os.path.exists(outPath):
		os.makedirs(outPath)

	if args.video_path!=0:
		countFileName = '{}/{}.csv'.format(outPath, args.video_path.split('/')[-1].split('.mp4')[0])
	else:
		countFileName = '{}/results_live_cam.csv'.format(outPath)
	countFile = open(countFileName, 'w')
	countFile.close()

print('Running on: {}'.format(device))


# Prepare detection model
mtcnn = MTCNN(keep_all=True, device=device, ov=args.ov)

# Load video
cap = cv2.VideoCapture(args.video_path)        
num_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.video_path!=0 else 10000

if args.le:
	skip_frames = 10
cumulative = 0
if args.le:
	skip_frames = 10
id=0 #fake (tracking) id
for fr in tqdm(range(num_images)):

	start = time.time()

	# Read frame
	res, frame = cap.read()

	if args.le and fr % skip_frames != 0:
		continue

	if fr % args.skip != 0:
		continue

	# Detect faces
	boxes, _ = mtcnn.detect(frame)
	end = time.time()

	if args.plot:
		frame_draw = frame.copy()
		if boxes is not None:
			for b in boxes:
				cv2.rectangle(frame_draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

		cv2.imshow('show', frame_draw)
		cv2.waitKey(1)

	if args.AVA:
		countFile = open(countFileName, 'a')
		txt = '{:.6f}'.format(end-start)
		if boxes is not None:
			for d in boxes:
				txt += ',-2,-2,-2,-2,{:.1f},{:.1f},{:.1f},{:.1f},{:d},-2,-2'.format(d[0], d[1], d[2], d[3], id)
				id+=1
		txt += '\n'
		countFile.write(txt)
		countFile.close()

	if args.video_path==0:
		if boxes is not None:
			for d in boxes:
				cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (0, 255, 0), 2)
		cv2.imshow('Live', frame)
		cv2.waitKey(1)
