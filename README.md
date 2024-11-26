# AUXILIARY STEERING SYSTEM FOR REDUCING TRAFFIC ACCIDENTS THROUGH COLLISION WARNING ALERT
![Imagem1](https://github.com/Gabriel-sci/FCW-project/assets/125495002/1df43293-93ed-4fab-976c-068b6c16234d)

# Abstract
This Project is part of a undergraduate research on DRIVING ASSISTANCE SYSTEM TO REDUCE TRAFFIC ACCIDENTS BY MEANS OF COLLISION AVOIDANCE ALERTS, promoted by Mackenzie Presbyterian University. The goal is to develop a low cost, embeeded collision warning system, with Raspberry Pi 4 using Yolov5 with the Pi camera, for object detection. To do so the relative speed (km/h) with other vehicles is calculated by the system.

# Instructions
To test the program, download the yolov5 required packages through the cmd, 
using this command: pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

The code developed is in the application.py file.

The trained yolov5 model was exported to .onnx file in order to achieve better fps performance on CPU, and it is available as model.onnx

The dataset used to train the yolov5 model network is available in roboflow: https://universe.roboflow.com/ic-8viu9/fcw_dataset
