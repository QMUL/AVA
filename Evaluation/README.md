# Evaluation code

The evaluation code enables the assessment of the quality of Anonymous Video Analytics algorithms.

## How to use?

- Activate AVA conda environment ([installation instructions](https://github.com/QMUL/AVA#installation)):
```
source activate AVA
```

- Evaluate localization, e.g.:
```
python localization.py --dataset_path='./AVA_dataset/' --estimation_path='./Airport-1.csv' --bodypart='face'
```

- Evaluate counting, e.g.:
```
python counting.py --dataset_path='./AVA_dataset/' --estimation_path='./Airport-1.csv'
```

- Evaluate age estimations, e.g.:
```
python age.py --dataset_path='./AVA_dataset/' --estimation_path='./Airport-1.csv'
```

- Evaluate gender estimations, e.g.:
```
python gender.py --dataset_path='./AVA_dataset/' --estimation_path='./Airport-1.csv'
```

Input arguments:
| Command | Description |
|--|--|
| -h | show a help message with all arguments and exit |
| --dataset_path | path to the AVA dataset |
| --estimation_path | path to a csv file in AVA format |
| --bodypart | body part to evaluate ('face' or 'person')  |

