# A3, DeepSORT

## Information
 - Publication tracker: [link](https://arxiv.org/abs/1602.00763)
 - Publication detector: [link](https://arxiv.org/abs/1804.02767)
 - Original code: [link](https://github.com/ZQPei/deep_sort_pytorch)

## Description

This is an implementation of the Multiple Object Tracking (MOT) algorithm deep sort. It is composed of a person detector (YOLOv3), tracking algorithm (SORT) with CNN feature extraction model 

## Install
- Download and uncompress PyTorch and OpenVINO model weights
```
wget http://ava.eecs.qmul.ac.uk/resources/weights/A3.zip
unzip A3.zip
```

## How to run
```
python run.py --video_path <pathToVideo> --AVA
```

## How to run (live camera)
```
python run.py --AVA
```

Main input arguments:
| Command | Description |
|--|--|
| -h | show a help message with all arguments and exit |
| --video_path | path to input video |
| --AVA | enables the output in AVA format |
| --ov | execution in CPU with OpenVINO optimization |
| --cpu | execution in CPU without OpenVINO optimization  |


## Output
More information [here](https://github.com/QMUL/AVA/tree/master/Analytics#output-data-format)
