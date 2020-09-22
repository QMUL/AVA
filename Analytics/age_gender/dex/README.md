
# A6, DEX: Deep EXpectation of apparent age from a single image

## Information
 - Publication: [link](https://ieeexplore.ieee.org/document/7406390#:~:text=Our%20proposed%20method%2C%20Deep%20EXpectation,images%20with%20apparent%20age%20annotations.)
 - Original code: [link](https://github.com/sajjjadayobi/FaceLib)

## Description
This is an implementation of DEX for age and gender estimation.

## Install

- Download and uncompress PyTorch and OpenVINO model weights
```
wget http://test-cyl.eecs.qmul.ac.uk/ava/A6.zip
unzip A6.zip
```

## How to run
```
python run.py --video_path <pathToVideo> --localization_path <pathToAVACSV> --AVA
```

Main input arguments:
| Command | Description |
|--|--|
| -h | show a help message with all arguments and exit |
| --video_path | path to input video |
| --localization_path | path to a csv file with face detections in AVA format. For example generated with A1 or A2 |
| --AVA | enables the output in AVA format |
| --ov | execution in CPU with OpenVINO optimization |
| --cpu | execution in CPU without OpenVINO optimization  |


## Output
More information [here](https://github.com/QMUL/AVA/tree/master/Analytics#output-data-format)

