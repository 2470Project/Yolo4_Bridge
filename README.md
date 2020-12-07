# Yolo4_Bridge
## Set up the virtual environment 
use conda or pip to install the packages for this project; recommend use conda if you try to
run the project on gpu. Some packages are unavailable for conda or pip; my suggestion is to 
try both alternatively.

## Preprocess data and get train and test data set
Use bridge_convert.py under scripts directory to generate train and test dataset called 
bridge.txt file under ./data/dataset .
In bridge_convert.py, the parameter to convert_annotation_to_dataset() is the number of 
photos of your dataset.
For every lines in bridge.txt, it consists of the path to img, 4 coordinates and the class 
index of every box.

## Train and apply transfer learning
### train from scratch 
python train.py
### apply transfer learning
python train.py --weights ./data/yolov4.weights
