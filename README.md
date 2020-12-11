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

train.py will train the model and save the weights to ./bridge/bridge

## Save Model
Save_model.py loads model weights from ./bridge and save the whole yolov4 model to ./checkpoints/bridge-416
python save_model.py --weights ./bridge --output ./checkpoints/bridge-416

## Detect image
Load model form ./checkpoints/bridge-416 and run on image
python detect_video.py --weights ./checkpoints/bridge-416 --video ./data/syc/wang1.mp4

## Detect video
Load model form ./checkpoints/bridge-416 and run on video
python detect_video.py --weights ./checkpoints/bridge-416 --video ./data/syc/wang1.mp4

## Detect vehicle and its direction
python yolo_tracker.py --weights ./checkpoints/bridge-416 --video ./data/syc/wang1.mp4

## Accuracy
Run test_scripts.py and accCount.py, and we get
The ratio of vehicles which are recognized by model is 0.61.
The accuracy of classification of recognized vehiclesia 0.97.


## Note
The images, tags, weights are all in google drive. And we should upload wang1.mp4 which is for detect_video. 
The model was fed by over 12K images after 5K images augmented and was trained on Nvidia 2080Ti GPU.


