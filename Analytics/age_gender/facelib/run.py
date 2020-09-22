from AgeGender.Detector import AgeGender
import cv2
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os

#Own libraries
#from libs.reader import *
import numpy as np
from libs.reader import estimationReader


# Parse arguments
parser = argparse.ArgumentParser(description='Age/Gender estimation')
parser.add_argument('--margin', type=float, default=0, help='Margin around detected face for age-gender estimation')
parser.add_argument('--video_path', type=str, default=None, help='Video path')
parser.add_argument('--localization_path', type=str, default=None, help='Localization output file path (it must be a face detector)')
parser.add_argument('--img_size', type=int, default=112)
#parser.add_argument('--system', type=str, default=None, required=True, help='System')
parser.add_argument('--plot', action='store_true', default=False, help='Enable age estimation')
parser.add_argument("--AVA", action="store_true", default=False, help='Enable output for AVA')
parser.add_argument('--ov', action='store_true', default=False, help='Run in openvino')
parser.add_argument('--cpu', action='store_true', default=False, help='Run with CPU')
args = parser.parse_args()

if args.AVA:
	if args.ov:
		outPath = '../../output/facelib_ov'
	elif args.cpu:
		outPath = '../../output/facelib_cpu'
	else:
		outPath = '../../output/facelib_gpu'
	if not os.path.exists(outPath):
		os.makedirs(outPath)

	ageFileName = '{}/{}.csv'.format(outPath, args.video_path.split('/')[-1].split('.mp4')[0])
	ageFile = open(ageFileName, 'w')
	ageFile.close()

if not args.ov:
	if args.cpu:
		device='cpu'	
	else:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
	device='cpu'

if device == 'cuda':
	cudnn.benchmark = True

print('Cuda/CPU? {}'.format(device))

# Read detections from retinaface file
#video_name = args.video_path.split('/')[-1].split('.mp4')[0]
#localization_path = '../../../localization_counting/output/{}/retinaface_{}/{}.txt'.format(args.system,'gpu' if device=='cuda' else 'ov', video_name)
detections = estimationReader(args.localization_path, bodypart='face')

age_gender_detector = AgeGender(name='full', weight_path='weights/facelib.pth', device=device, ov=args.ov)

cap = cv2.VideoCapture(args.video_path)
num_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with torch.no_grad():
	for fr in tqdm(range(num_images)):
		start = time.time()
	
		ret, frame = cap.read()
		img_h, img_w, _ = np.shape(frame)
		
		# Detect (read) faces in the current frame, if the frame should be (was) processed
		if fr in detections.data:
			bboxes = detections.data[fr]['bboxes']
		else:
			continue

		#faces, boxes, scores, landmarks = face_detector.detect_align(frame)
		if len(bboxes):
			faces = np.empty((len(bboxes), args.img_size, args.img_size, 3))
			for i, box in enumerate(bboxes):
				x1, y1, x2, y2, w, h = box[0], box[1], box[2]+1, box[3]+1, box[2]-box[0], box[3]-box[1]
				xw1 = max(int(x1 - args.margin * w), 0)
				yw1 = max(int(y1 - args.margin * h), 0)
				xw2 = min(int(x2 + args.margin * w), img_w - 1)
				yw2 = min(int(y2 + args.margin * h), img_h - 1)
				faces[i] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1], (args.img_size, args.img_size))
			faces = torch.from_numpy(faces)
			
			genders, ages = age_gender_detector.detect(faces)

			if args.plot:
				for i, b in enumerate(bboxes):
					cv2.putText(frame, '{},{}'.format(genders[i],ages[i]), (int(b[0]), int(b[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, [0, 200, 0], 3)
					cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 3)

				cv2.imshow('frame', frame)
				if cv2.waitKey(1) == ord('q'):
					break
		
		if args.AVA:
			end = time.time()

			ageFile = open(ageFileName, 'a')
			txt = '{:.6f}'.format(end-start)
			for i, d in enumerate(bboxes):
				txt += ',-2,-2,-2,-2,{:.1f},{:.1f},{:.1f},{:.1f},{:d},{:d},{:d}'.format(d[0], d[1], d[2], d[3], detections.data[fr]['ids'][i], ages[i], genders[i])
			txt += '\n'
			ageFile.write(txt)
			ageFile.close()
