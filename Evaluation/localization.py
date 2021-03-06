from tqdm import tqdm 
import argparse
from tabulate import tabulate
import os

import numpy as np

# Own libraries
from libs.reader import *
from libs.metrics import *

#sample call: python localization.py --dataset_path='./AVA_dataset/' --estimation_path='./Airport-1.csv' --fps=30 --bodypart='face'

def parse_args():
	parser = argparse.ArgumentParser(description='BAVA localization evaluation')
	parser.add_argument('--dataset_path', required=True, type=str, help='Path to the dataset directory')
	parser.add_argument('--estimation_path', required=True, type=str, help='Path to the localization estimation file generated by a BAVA localization algorithm')
	parser.add_argument('--fps', required=False, type=int, help='Frame rate of the videos used as input (e.g. 30)', choices=[30,3,1])
	parser.add_argument('--bodypart', required=True, type=str, help='Body part to evaluate (e.g. face)', choices=['person','face'])
	return parser.parse_args()

if __name__== "__main__":

	args = parse_args()
	args.fps = 30

	outdir = 'out/localization'
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	video_name = args.estimation_path.split('/')[-1].split('.csv')[0]
	step = int(30/args.fps)
	file = open('out/localization/eval_{}.txt'.format(video_name), 'w') 
	
	Ps = []
	Rs = []
	Fs = []
	R_close = []
	R_far = []
	R_occ0 = []
	R_occ1 = []
	R_occ2 = []
	times = []
	
	print('Evaluating localization in {} at {}fps on {}...'.format(video_name, args.fps, args.bodypart))
	
	# Read annotation file
	GT = annotationReader('{}{}.xml'.format(args.dataset_path, video_name), mode=args.bodypart)

	# Read estimation file
	EST = estimationReader(args.estimation_path, bodypart=args.bodypart)

	# Compute metrics
	TP_FP_FN=[]
	TP_FP_FN_distance_close=[]
	TP_FP_FN_distance_far=[]
	TP_FP_FN_occlusion0=[]
	TP_FP_FN_occlusion1=[]
	TP_FP_FN_occlusion2=[]
	last_frame = max(GT.last_frame, EST.last_frame)

	for fr in tqdm(range(0,last_frame, step)):
		try:
			TP_FP_FN.append(computeTP_FP_FN(GT, EST, fr, mode=args.bodypart))
			TP_FP_FN_distance = computeTP_FP_FN_distance(GT, EST, fr, mode=args.bodypart)
			TP_FP_FN_distance_close.append(TP_FP_FN_distance[0])
			TP_FP_FN_distance_far.append(TP_FP_FN_distance[1])
			TP_FP_FN_occlusion = computeTP_FP_FN_occlusion(GT, EST, fr, mode=args.bodypart)
			TP_FP_FN_occlusion0.append(TP_FP_FN_occlusion[0])
			TP_FP_FN_occlusion1.append(TP_FP_FN_occlusion[1])
			TP_FP_FN_occlusion2.append(TP_FP_FN_occlusion[2])
		except:
			print('Error in frame {}, continuing'.format(fr))
			continue

	TP_FP_FN = np.sum(TP_FP_FN,axis=0)
	TP_FP_FN_distance_close = np.sum(TP_FP_FN_distance_close,axis=0)
	TP_FP_FN_distance_far = np.sum(TP_FP_FN_distance_far,axis=0)
	TP_FP_FN_occlusion0 = np.sum(TP_FP_FN_occlusion0,axis=0)
	TP_FP_FN_occlusion1 = np.sum(TP_FP_FN_occlusion1,axis=0)
	TP_FP_FN_occlusion2 = np.sum(TP_FP_FN_occlusion2,axis=0)

	#Calculate P, R, F
	#Global
	TP, FP, FN = TP_FP_FN
	P=(TP/(TP+FP))
	R=(TP/(TP+FN))
	F=((2*P*R)/(P+R))

	# Distance
	TP, FP, FN = TP_FP_FN_distance_close
	Rdistance_close=(TP/(TP+FN))
	#
	TP, FP, FN = TP_FP_FN_distance_far
	Rdistance_far=(TP/(TP+FN))

	# Occlusions
	TP, FP, FN = TP_FP_FN_occlusion0
	Rocclusion0=(TP/(TP+FN))
	#
	TP, FP, FN = TP_FP_FN_occlusion1
	Rocclusion1=(TP/(TP+FN))
	#
	TP, FP, FN = TP_FP_FN_occlusion2
	Rocclusion2=(TP/(TP+FN))

	# Print out results in a txt file
	times = np.array([EST.data[frame]['time'] for frame in EST.data])
	table = tabulate([ ['{:.2f}'.format(P), '{:.2f}'.format(R), '{:.2f}'.format(F), '{:.2f}'.format(Rdistance_close), '{:.2f}'.format(Rdistance_far), '{:.2f}'.format(Rocclusion0), '{:.2f}'.format(Rocclusion1), '{:.2f}'.format(Rocclusion2), '{:.4f} +- {:.4f}'.format(np.mean(times),np.std(times))]], headers=['Precision','Recall','F1-Score','Recall-close','Recall-far','Recall-no-occlussion','Recall-partial-occlussion','Recall-heavy-occlussion','Execution time [s]'])
	file.write(table)
	file.close()