
# A1, RetinaFace: Single-stage Dense Face Localisation in the Wild

## Information
 - Publication: [link](https://arxiv.org/abs/1905.00641)
 - Original code: [link](https://github.com/biubug6/Pytorch_Retinaface)

## Description

This is an implement of RetinaFace: Single-stage Dense Face Localisation in the Wild. A robust single-stage face detector that performs pixel-wise face localization on various scales of faces by taking advantage of joint extra-supervised and self-supervised multi-task learning.

## Install
- Download and uncompress PyTorch and OpenVINO model weights
```
wget http://ava.eecs.qmul.ac.uk/resources/weights/A1.zip
unzip A1.zip
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
