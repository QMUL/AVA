#!/bin/bash
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

cd ..