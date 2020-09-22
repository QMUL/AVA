
# A5, FaceLib

## Information
 - Original code: [link](https://github.com/sajjjadayobi/FaceLib)

## Description
This is an implementation of FaceLib for age and gender estimation.

## Install

- Download and uncompress PyTorch and OpenVINO model weights
```
wget http://ava.eecs.qmul.ac.uk/resources/weights/A5.zip
unzip A5.zip
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

