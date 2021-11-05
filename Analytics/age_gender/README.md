# Age and gender estimation algorithms

Age and gender estimation algorithms require as input a crop with the person's face. For running age/gender estimation algorithms it is required to first generate localization results by a face detector (Algorithm 1 or Algorithm 2). Then, the localization output file is used as input for the age/gender estimation.

## Age and gender estimation:
The implemented age and gender estimation algorithms are:
- **Algorithm 5:** [FaceLib](https://github.com/QMUL/AVA/tree/master/Analytics/age_gender/facelib)
- **Algorithm 6:** [DEX](https://github.com/QMUL/AVA/tree/master/Analytics/age_gender/dex)

Both algorithms estimate simultaneously age and gender.

## Output data format
The output data format is as indicated [here](https://github.com/QMUL/AVA/tree/master/Analytics#output-ava-data-format). The -2 (unavailable) results by localization algorithms will be filled by the estimated age and gender.
