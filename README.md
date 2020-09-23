# Benchmark for Anonymous Video Analytics (AVA)

<p align="center">
<img src="http://ava.eecs.qmul.ac.uk/resources/intro_github2.jpg" alt="Annoymous video analytics framework" style="height:200px"/>
</p>

Welcome to the official site of <i>Benchmark for Anonymous Video Analytics</i> (AVA) for digital out-of-home audience measurement.
AVA aims to enable real-time understanding of audiences exposed to advertisements in order to estimate the reach and effectiveness of each advertisement.
     
AVA relies on person detectors or trackers to localize people and to enable the estimation of audience attributes, such as their demographics. The benchmark is composed of:
-   a set of performance scores specifically designed for audience measurement and an evaluation tool;
-   a novel fully-annotated dataset for digital out-of-home AVA;
-   open-source codes including detectors, trackers, and age and gender estimation baseline algorithms, evaluation codes; and 
-   benchmarking of the baseline algorithms and two commercial off-the-shelf solutions in real-world on-the-edge settings.

## The benchmark
The benchmark considers localization, count, age, and gender as attributes for the analytics. The taxonomy is depicted in the figure below. AVA algorithms should ensure the preservation of the privacy of audience members by performing inferences and aggregating them directly on edge systems, without recording or streaming raw data.
     
<p align="center">
 <img src="http://ava.eecs.qmul.ac.uk/resources/taxonomy.jpg" width="800" alt="Taxonomy">
</p>

We consider a person to have <i>Opportunity to See</i> (OTS) the signage when their face is visible from the left profile to the right profile, and the person is <i>not</i> heading opposite to the location of the camera, as shown in the figure below. We consider only the attributes of people with OTS.

<p align="center">
 <img src="http://ava.eecs.qmul.ac.uk/resources/OTS.jpg" width="800" alt="Opportunity to see">
</p>


## The dataset
The dataset was collected in settings that mimic real-world signage-camera setups used for AVA, and it is composed of 16 videos recorded at different locations such as airports, malls, subway stations, and pedestrian areas. Outdoor videos are recorded at different day times such as morning, afternoon, and evening.
The dataset is recorded with Internet Protocol or USB fixed cameras with wide and narrow lenses to mimic the real-world use cases. Videos are recorded at 1920x1080 resolution and 30fps.
The dataset includes videos of duration between 2 minutes and 30 seconds, and 6 minutes and 26 seconds, totaling over 78 minutes, with over 141,000 frames.
The videos feature 34 professional actors. The dataset includes multi-ethnic people including Asian, Caucasian, African-American, and Hispanic; whose age ranges from 10 to 80, including male and female genders. Also, people have been recorded with varied emotions while looking at the signage.
A sample frame of each location is shown below. For the <i>mall</i> location, two videos are recorded at different temporal moments: indoors (Mall-1/2) and outdoors (Mall-3/4).

<table class="tg2">
<tbody>
<tr>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Airport-1</span></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Airport-2</span></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Airport-3</span></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Airport-4</span></th>
</tr>
<tr>
<th class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/1. Airport- 24Mm- 500 Lux.jpg" alt="Airport-1" width="250"></th>
<th class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/3. Airport- 35Mm- 500 Lux-20.jpg" alt="Airport-1" width="250"></th>
<th class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/9. Airport- 35Mm- 600 Lux-21.jpg" alt="Airport-1" width="250"></th>
<th class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/14. Airport- 35Mm- 200-500 Lux-13.jpg" alt="Airport-1" width="250"></th>
</tr>
<tr>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Mall-1/2</span></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Mall-3/4</span></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Pedestrian-1</span></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Pedestrian-2</span></th>
</tr>
<tr>
<td class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/1. Mall- 24Mm- 300 Lux-2.jpg" alt="Airport-1" width="250"></td>
<td class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/10. Mall- 24Mm- 800 Lux-8.jpg" alt="Airport-1" width="250"></td>
<td class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/2. Pedestrian - 2- Afternoon-60K Lux – 24Mm-14.jpg" alt="Airport-1" width="250"></td>
<td class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/7. Pedestrian - 2- Afternoon-40K Lux– 35Mm-20.jpg" alt="Airport-1" width="250"></td>
</tr>
<tr>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Pedestrian-3</span></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Pedestrian-4</span></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Pedestrian-5</span></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Subway-1</span></th>
</tr>
<tr>
<td class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/6. Pedestrian –5 –Midday-Overcast- 7000 Lux – 35mm.jpg" alt="Airport-1" width="250"></td>
<td class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/8. Pedestrian – 7- Daytime Shade- 5500 Lux – 35mm.jpg" alt="Airport-1" width="250"></td>
<td class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/11. Pedestrian – 3+4- Evening + Night- 250 Lux- 24Mm-10.jpg" alt="Airport-1" width="250"></td>
<td class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/1. Subway- 180 Lux- 35Mm-3.jpg" alt="Airport-1" width="250"></td>
</tr>
<tr>
<th class="tg2-6uvz"></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Subway-2</span></th>
<th class="tg2-6uvz"><span style="font-size:14.0pt">Subway-3</span></th>
<th class="tg2-6uvz"></th>
</tr>
<tr>
<td class="tg2-wk8r"></td>
<td class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/16. Subway- 24Mm- 180 Lux-28.jpg" alt="Airport-1" width="250"></td>
<td class="tg2-wk8r"><img src="http://ava.eecs.qmul.ac.uk/resources/dataset/13. Subway- 35Mm- 200 Lux-19.jpg" alt="Airport-1" width="250"></td>
<td class="tg2-wk8r"></td>
</tr>
</tbody>
</table>


## Installation
The algorithms have been tested on Ubuntu 16.04 and MacOS 10.15.6 with the libraries on the version requested in requirements.txt and with OpenVINO 2020.1.

The creation of a single conda environment suffices for running all baseline algorithms and evaluation codes present in the repository.

 - (Optional) Install OpenVINO 2020.1 [link](https://docs.openvinotoolkit.org/latest/index.html). This is only required if CPU OpenVINO optimization is wanted. 
 
 - Download and uncompress AVA dataset [1.31 GB]
 ```
 wget http://ava.eecs.qmul.ac.uk/resources/AVA_dataset.zip
 unzip AVA_dataset.zip
```
 
 - Install miniconda: [link](https://docs.conda.io/en/latest/miniconda.html)

 - Create conda environment with Python 3.7
```
conda create -n AVA python=3.7
```

 - Activate AVA conda environment:
```
source activate AVA
```

- Clone the AVA repository
```
git clone https://github.com/QMUL/AVA.git
```

 - Install requirements
```
pip install -r requirements.txt
```

- Download the desired model weigths, for all algorithms
```
bash get_all_weights.sh
```


## Related material
 - Paper: pre-print (coming soon) and publication (under review) 
 - [Project website](http://ava.eecs.qmul.ac.uk/) 
 - [Dataset download](http://ava.eecs.qmul.ac.uk/Dataset) 
 - [Evaluation tool](http://ava.eecs.qmul.ac.uk/Evaluation)

## Metadata
- Authors: [Ricardo Sanchez-Matilla](mailto:ricardo.sanchezmatilla@qmul.ac.uk) and [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk)
- Created date: 21/09/2020
- Version: 0.1
- Resource type: software


## License
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
