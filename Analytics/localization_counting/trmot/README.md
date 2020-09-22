
# A4, Towards-realtime-MOT: Joint Detection and Embedding for fast multi-object tracking

## Information
 - Publication: [link](https://arxiv.org/abs/1909.12605)
 - Original code: [link](https://github.com/Zhongdao/Towards-Realtime-MOT)

## Description
This is an implementation of Joint Detection and Embedding (JDE) model. JDE is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simultaneously in a shared neural network

## Install
- Download and uncompress PyTorch model weights (OpenVINO model is not available)
```
wget http://ava.eecs.qmul.ac.uk/resources/weights/A4.zip
unzip A4.zip
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
| --cpu | execution in CPU without OpenVINO optimization  |


## Output
More information [here](https://github.com/QMUL/AVA/tree/master/Analytics#output-data-format)
