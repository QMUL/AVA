#!/bin/bash

#Action! To add the path to dataset (e.g. /home/myname/Dataset/*.mp4)
VIDEOS=

#Action! To add the path to conda activate (e.g. "/home/myname/miniconda3/bin/activate")
source ""
root=`pwd`
clear

conda activate AVA


#####################
##  A1: RetinaFace ##
#####################
cd $root
cd retinaface
for f in $VIDEOS
do
	echo "RetinaFace: processing $f file..."
	python run.py --video_path "$f" --AVA --cpu
done

#################
##  A2: MTCNN  ##
#################
cd $root
cd mtcnn
for f in $VIDEOS
do
	echo "MTCNN: processing $f file..."
	python run.py --video_path "$f" --AVA --cpu
done

####################
## A3: Deep Sort  ##
####################
cd $root
cd deepsort
for f in $VIDEOS
do
	echo "DeepSort: processing $f file..."
	python run.py --video_path "$f" --AVA --cpu
done

#################
## A4: TRMOT   ##
#################
cd $root
cd trmot
for f in $VIDEOS
do
	echo "TRMOT: processing $f file..."
	python run.py --video_path "$f" --AVA --cpu
done

