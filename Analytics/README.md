# Baseline algorithms

This repository is composed of 6 baseline algorithms, that are divided into two categories: localization, and age and gender estimation.

## Localization
Localization algorithms are the basis of AVA analytics. Localization algorithms, such as detectors or trackers, allow one to detect the presence and location of people on a video. Once the people have been localized, further audience attributes such as age and gender can be estimated. Also, localization algorithms enable people counting. 

-[**Localization algorithms**](https://github.com/QMUL/AVA/tree/master/Analytics/localization_counting/)

## Age and gender
Age and gender estimation algorithms require as input a crop with the person's face. For running age/gender estimation algorithms it is required to first generate localization results. Then, the localization output file is used as input for the age/gender estimation algorithm.

-[**Age/gender estimation algorithms**](https://github.com/QMUL/AVA/tree/master/Analytics/age_gender/)


# Output AVA data format
All baseline algorithms generate the estimations in CSV files with the following data format.

For each video, the algorithm outputs a single CSV file. The CSV file has <i>T</i> rows, where <i>T</i> is the number of frames in the video, and where the columns represent:

<p align="justify"> 
<ul>
	<li><i>time</i>: float that indicates the time, in seconds, the algorithm took to generate the results for the current frame</li>
	<li><i>person_j</i>: eight floats (or integers) that indicate the x0, y0, x1, y1 (x0, y0, x1, y1) coordinates of the bounding box (the origin of the reference system is the top-left corner of the frame) of the j-th <i>person</i> (<i>face</i>) with Opportunity to See (OTS) the signage, as defined in the paper, detected by the algorithm in <i>frame</i>,</li>
<ul>
	<li><i>x0</i>: horizontal coordinate of the top-left corner of the person/face</li>
	<li><i>y0</i>: vertical coordinate of the top-left corner of the person/face</li>
	<li><i>x1</i>: horizontal coordinate of the bottom-right corner of the person/face</li>
	<li><i>y1</i>: vertical coordinate of the bottom-right corner of the person/face</li>
</ul>
	<li>three integers that indicate the values of the identity/age/gender attributes of the j-th person</li>
	<ul>
		<li><i>id</i>: integer that indicates the person identity</li>
		<li><i>age</i>: integer that indicates the estimated person age</li>
		<li><i>gender</i>: 0=<i>male</i>; 1=<i>female</i></li>
	</ul>
</ul>

<p align="justify"> 
	<img src="http://www.eecs.qmul.ac.uk/~rsm31/ava/resources/data_format.jpg" alt="Data format" width="800px"/>
</p>

<p align="justify"> 
For localization (face or person), identity, age, and gender the outputs shall be <i>-1</i> if the estimation (output) is unknown (e.g. the face is not visible); or <i>-2</i> if the algorithm does not provide an output for that attribute.	
</p>

This <a href="http://www.eecs.qmul.ac.uk/~rsm31/ava/resources/data_format.csv" targer = "_blank"><img src="http://www.eecs.qmul.ac.uk/~rsm31/ava/resources/csv.png" alt="resultsPDF" width="16px"/> <b>CSV file</b></a> is an example of output file for a sample 30-frame video, <i>T</i>=30, with no person detected until frame 6, a person (with identity 0) detected from frame 6 onwards, and two people (with identity 0 and 1) detected from frame 20 onwards.

