from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

from tqdm import tqdm

import time

from inference import load_to_IE, sync_inference, async_inference, get_async_output

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
					type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--video_path', type=str, required=False, default=0, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument("--AVA", action="store_true", default=False, help='Enable output for AVA')
parser.add_argument("--ov", action="store_true", default=False, help='Run in openvino')
parser.add_argument("--le", action="store_true", default=False, help='Low end mode')
parser.add_argument('--skip', default=0, type=int, help='skip X frames')

args = parser.parse_args()

if args.AVA:
	if args.ov:
		if args.skip:
			outPath = '../../output/retinaface_ov/{}'.format(args.skip)
		else:
			outPath = '../../output/retinaface_ov'
	elif args.cpu:
		if args.skip:
			outPath = '../../output/retinaface_cpu/{}'.format(args.skip)
		else:
			outPath = '../../output/retinaface_cpu'
	else:
		if args.skip:
			outPath = '../../output/retinaface_gpu/{}'.format(args.skip)
		else:
			outPath = '../../output/retinaface_gpu'
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


def check_keys(model, pretrained_state_dict):
	ckpt_keys = set(pretrained_state_dict.keys())
	model_keys = set(model.state_dict().keys())
	used_pretrained_keys = model_keys & ckpt_keys
	unused_pretrained_keys = ckpt_keys - model_keys
	missing_keys = model_keys - ckpt_keys
	assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
	return True


def remove_prefix(state_dict, prefix):
	''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
	f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
	return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
	if load_to_cpu:
		pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
	else:
		device = torch.cuda.current_device()
		pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
	if "state_dict" in pretrained_dict.keys():
		pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
	else:
		pretrained_dict = remove_prefix(pretrained_dict, 'module.')
	check_keys(model, pretrained_dict)
	model.load_state_dict(pretrained_dict, strict=False)
	return model


if __name__ == '__main__':
	torch.set_grad_enabled(False)

	cfg = None
	if args.network == "mobile0.25":
		cfg = cfg_mnet
	elif args.network == "resnet50":
		cfg = cfg_re50

	if not args.ov:
		# net and model
		net = RetinaFace(cfg=cfg, phase = 'test')
		net = load_model(net, args.trained_model, args.cpu)
		net.eval()
		device = torch.device("cpu" if args.cpu else "cuda")
		net = net.to(device)
		print('Device is {}'.format(device))
	else:
		net = load_to_IE('./weights/FaceDetector.xml')
		device = torch.device("cpu")

	cudnn.benchmark = True
	
	cap = cv2.VideoCapture(args.video_path)        
	num_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.video_path!=0 else 10000
	if args.le:
		skip_frames = 10
	id=0 #fake (tracking) id
	for fr in tqdm(range(num_images)):
		start = time.time()

		# Read frame
		res, img_raw = cap.read()

		if args.le and  fr % skip_frames != 0:
			continue

		if fr % args.skip != 0:
			continue
			
		if args.ov:
			img = cv2.resize(img_raw, (640,640))
			img = np.float32(img)
		else:
			img = np.float32(img_raw)

		# testing scale
		target_size = 1600
		max_size = 2150
		im_shape = img.shape
		im_size_min = np.min(im_shape[0:2])
		im_size_max = np.max(im_shape[0:2])
		resize = float(target_size) / float(im_size_min)
		# prevent bigger axis from being more than max_size:
		if np.round(resize * im_size_max) > max_size:
			resize = float(max_size) / float(im_size_max)
		if args.origin_size:
			resize = 1

		if resize != 1:
			img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
		im_height, im_width, _ = img.shape
		scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
		img -= (104, 117, 123)
		img = img.transpose(2, 0, 1)
		img = torch.from_numpy(img).unsqueeze(0)
		img = img.to(device)
		scale = scale.to(device)


		if not args.ov:
			loc, conf, landms = net(img)  # forward pass
		else:
			result = sync_inference(net, image=img)
			loc = torch.from_numpy(result['Concat_289'])
			conf = torch.from_numpy(result['Softmax_340'])
			landms = torch.from_numpy(result['Concat_339'])
		
		priorbox = PriorBox(cfg, image_size=(im_height, im_width))
		priors = priorbox.forward()
		priors = priors.to(device)
		prior_data = priors.data
		boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
		boxes = boxes * scale / resize
		boxes = boxes.cpu().numpy()
		scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
		landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
		scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
							   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
							   img.shape[3], img.shape[2]])
		scale1 = scale1.to(device)
		landms = landms * scale1 / resize
		landms = landms.cpu().numpy()

		# ignore low scores
		inds = np.where(scores > args.confidence_threshold)[0]
		boxes = boxes[inds]
		landms = landms[inds]
		scores = scores[inds]

		# keep top-K before NMS
		order = scores.argsort()[::-1]
		# order = scores.argsort()[::-1][:args.top_k]
		boxes = boxes[order]
		landms = landms[order]
		scores = scores[order]

		# do NMS
		dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
		keep = py_cpu_nms(dets, args.nms_threshold)
		# keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
		dets = dets[keep, :]
		landms = landms[keep]

		# keep top-K faster NMS
		# dets = dets[:args.keep_top_k, :]
		# landms = landms[:args.keep_top_k, :]

		dets = np.concatenate((dets, landms), axis=1)
		end = time.time()

		if args.ov:
			ratio_x = 1920./640
			ratio_y = 1080./640
			dets[:,[0,2,5,7,9,11,13]] = dets[:,[0,2,5,7,9,11,13]]*ratio_x
			dets[:,[1,3,6,8,10,12,14]] = dets[:,[1,3,6,8,10,12,14]]*ratio_y

		# save image
		if args.save_image:
			for b in dets:
				if b[4] < args.vis_thres:
					continue
				text = "{:.4f}".format(b[4])
				b = list(map(int, b))
				cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
				cx = b[0]
				cy = b[1] + 12
				cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

				# landms
				cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
				cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
				cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
				cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
				cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

			# save image
			if not os.path.exists("./results/"):
				os.makedirs("./results/")
			name = "./results/" + str(fr) + ".jpg"
			cv2.imwrite(name, img_raw)

		if args.AVA:
			countFile = open(countFileName, 'a')
			txt = '{:.6f}'.format(end-start)
			for d in dets:
				txt += ',-2,-2,-2,-2,{:.1f},{:.1f},{:.1f},{:.1f},{:d},-2,-2'.format(d[0], d[1], d[2], d[3], id)
				id+=1
			txt += '\n'
			countFile.write(txt)
			countFile.close()

		if args.video_path==0:
			for d in dets:
				if d[4] < args.vis_thres:
					continue
				cv2.rectangle(img_raw, (d[0], d[1]), (d[2], d[3]), (0, 255, 0), 2)
			cv2.imshow('Live', img_raw)
			cv2.waitKey(1)
