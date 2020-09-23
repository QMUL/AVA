# Localization algorithms

Anonymous Video Analytics algorithms rely on localization algorithms, such as detectors or trackers, that allow one to detect the presence and location of people on a video.
A detector aims to localize objects of interest (e.g. people) given an input signal at a specific moment in time (e.g. frame).
A tracker aims to localize objects of interest given an input signal for a sequence of moments in time (e.g. video) and to assign an identity to each object so that it is unequivocally identified over time.

## Detection-based localization:
The implemented detectors localize faces.

- **Algorithm 1:** [RetinaFace](https://github.com/QMUL/AVA/tree/master/Analytics/localization_counting/retinaface)
- **Algorithm 2:** [MTCNN](https://github.com/QMUL/AVA/tree/master/Analytics/localization_counting/mtcnn)

## Tracking-based localization:
The implemented trackers localize full person bodies.

- **Algorithm 3:** [DeepSort](https://github.com/QMUL/AVA/tree/master/Analytics/localization_counting/deepsort)
- **Algorithm 4:** [TRMOT](https://github.com/QMUL/AVA/tree/master/Analytics/localization_counting/trmot)

# Running on the whole dataset
We provide a script that it can be used to run an/some algorithm(s) in the whole dataset.

- Prior to run th script, fill path to dataset and conda in <i>run.sh</i>. Then, for GPU inference
```
bash runGPU.sh
```

or, for CPU inference
```
bash runCPU.sh
```
# Output data format
The output data format is as indicated [here](https://github.com/QMUL/AVAB/tree/master/Analytics#output-data-format). Age and gender spaces are filled by -2, as localization algorithms do not provide such as information.
