# ResNet18 + BiGRU
ResNet18+GRU is a model-free gait recognition method that operates with GEI silhouettes. This model is scalable and defines a baseline DL spatio-temporal model that requires improvements to view changes.

## Prerequisites
Python 3.9
GPU with CUDA (12.*)
PyTorch (2.5.1+cu121)
Torchvision (0.20.1+cu121)
Opencv-python 
NumPy 
Scikit-learn 

## Dataset & Preparation
- Step 1: Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
- Step 2: Set Dataset Path in: data_dir = r"path_to_CASIAB" # Change to CASIA-B Path

## Train
```bash
python gait_recogniton.py
```
- Run the Model and select '1'. The model path will be saved in the same directory
- When testing the model in a video mention its location: video_path = r"Video_Path"  #Add video Path
