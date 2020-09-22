#!/bin/bash

#Action! To add the path to dataset (e.g. /home/myname/Dataset/*.mp4)
VIDEOS=

#Action! To add the path to conda activate (e.g. "/home/myname/miniconda3/bin/activate")
source ""
root=`pwd`
clear

conda activate AVA

#################
## A5: FaceLib ##
#################
cd $root
cd facelib
for f in $VIDEOS
do
	echo "FaceLib: processing $f file..."
	python run.py --video_path "$f" --AVA --localization_path="TO FILL"
done


#################
##  A6: DEX    ##
#################
cd $root
cd dex
for f in $VIDEOS
do
	echo "DEX: processing $f file..."
	python run.py --video_path "$f" --AVA --localization_path="TO FILL"
done

#localization_path should be an AVA output file from a face detection algorithm (e.g. A1 or A2)
