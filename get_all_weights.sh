#!/bin/bash
cd Analytics

#Localization counting algorithms
cd localization_counting

cd retinaface
wget http://test-cyl.eecs.qmul.ac.uk/ava/A1.zip
unzip A1.zip
rm A1.zip

cd ../mtcnn
wget http://test-cyl.eecs.qmul.ac.uk/ava/A2.zip
unzip A2.zip
rm A2.zip

cd ../deepsort
wget http://test-cyl.eecs.qmul.ac.uk/ava/A3.zip
unzip A3.zip
rm A3.zip

cd ../trmot
wget http://test-cyl.eecs.qmul.ac.uk/ava/A4.zip
unzip A4.zip
rm A4.zip


#Age and gender algorithms
cd ../../age_gender

cd facelib
wget http://test-cyl.eecs.qmul.ac.uk/ava/A5.zip
unzip A5.zip
rm A5.zip

cd ../dex
wget http://test-cyl.eecs.qmul.ac.uk/ava/A6.zip
unzip A6.zip
rm A6.zip


cd ../../..