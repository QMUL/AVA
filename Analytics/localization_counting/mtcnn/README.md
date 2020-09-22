# A2, MTCNN: Multi-task Cascaded Convolutional Networks

## Information
 - Publication: [link](https://arxiv.org/pdf/1503.03832)
 - Original code: [link](https://github.com/timesler/facenet-pytorch)

## Description
This is an implementation of MTCNN: Multi-task Cascaded Convolutional Networks.

## Install
- Download and uncompress PyTorch and OpenVINO model weights
```
wget http://test-cyl.eecs.qmul.ac.uk/ava/A2.zip
unzip A2.zip
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
