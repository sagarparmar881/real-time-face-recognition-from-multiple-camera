Face recognition from camera with Dlib
######################################

Introduction
************

Detect and recognize single or multi faces from camera;

[1] Face register GUI with Tkinter, support setting (chinese) name when registering

### images-01

[2] Simple face register GUI with OpenCV, tkinter not needed and cannot set name

### images-02

[3] Too close to the camera, or face ROI out of camera area, will have "OUT OF RANGE" warning.

### images-03

[4] Generate face database from images captured
[5] Face recognizer
   

face_reco_from_camera_ot.py. Use OT to instead of re-reco for every frame to improve FPS:

### images-03


About accuracy:

* When using a distance threshold of ``0.6``, the dlib model obtains an accuracy of ``99.38%`` on the standard LFW face recognition benchmark.

About algorithm

* Residual Neural Network/ CNN

* This model is a ResNet network with 29 conv layers.
It's essentially a version of the ResNet-34 network from the paper Deep Residual Learning for Image Recognition
by He, Zhang, Ren, and Sun with a few layers removed and the number of filters per layer reduced by half.

Overview
********

(no OT) / 
Design of this repo, do detection and recognization for every frame:

.. image:: introduction/overview.png

(with OT) / OT used:

.. image:: introduction/overview_with_ot.png


Use OT can save the time for face descriptor computation to improve FPS. 

Steps
*****

## Git clone source code

      git clone https://github.com/coneypo/Dlib_face_recognition_from_camera

## Install some python packages needed

      pip install -r requirements.txt

## Tkinter GUI / Register faces with Tkinter GUI
      # Install Tkinter
      sudo apt-get install python3-tk python3-pil python3-pil.imagetk

      python3 get_faces_from_camera_tkinter.py

## OpenCV GUI / Register faces with OpenCV GUI, same with above step

      python3 get_face_from_camera.py

## Features extraction and save into ``features_all.csv``

      python3 features_extraction_to_csv.py

## Real-time face recognition

      python3 face_reco_from_camera.py

## Real-time face recognition (Better FPS compared with ``face_reco_from_camera.py``)

      python3 face_reco_from_camera_single_face.py

## Real-time face recognition with OT (Better FPS)

      python3 face_reco_from_camera_ot.py

About Source Code
*****************
Code structure:

    ├── get_faces_from_camera.py        		# Step 1. Face register GUI with OpenCV
    ├── get_faces_from_camera_tkinter.py                # Step 1. Face register GUI with Tkinter
    ├── features_extraction_to_csv.py   		# Step 2. Feature extraction
    ├── face_reco_from_camera.py        		# Step 3. Face recognizer
    ├── face_reco_from_camera_single_face.py            # Step 3. Face recognizer for single person
    ├── face_reco_from_camera_ot.py                     # Step 3. Face recognizer with OT
    ├── face_descriptor_from_camera.py  		# Face descriptor computation
    ├── how_to_use_camera.py            		# Use the default camera by opencv
    ├── data
    │   ├── data_dlib        			        # Dlib's model
    │   │   ├── dlib_face_recognition_resnet_model_v1.dat
    │   │   └── shape_predictor_68_face_landmarks.dat
    │   ├── data_faces_from_camera                      # Face images captured from camera (will generate after step 1)
    │   │   ├── person_1
    │   │   │   ├── img_face_1.jpg
    │   │   │   └── img_face_2.jpg
    │   │   └── person_2
    │   │       └── img_face_1.jpg
    │   │       └── img_face_2.jpg
    │   └── features_all.csv            	        # CSV to save all the features of known faces (will generate after step 2)
    ├── README.rst
    └── requirements.txt                		# Some python packages needed

# More 

#. Dlib Python api You can refer to this link for more information of how to use dlib: http://dlib.net/python/index.html

* Blog: https://www.cnblogs.com/AdaminXie/p/9010298.html

* Blog: https://www.cnblogs.com/AdaminXie/p/13566269.html

* Feel free to create issue or contribute PR for it:)

Thanks for your support.
