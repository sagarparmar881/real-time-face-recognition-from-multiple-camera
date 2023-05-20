
# Face Recognition from Multiple Camera with Dlib

* Project Goal: The goal of this project is to develop a system that can recognize faces from multiple cameras in real time.

* Project Approach: The system will use the Dlib library to detect and recognize faces. Dlib is a C++ library with Python bindings that provides a number of tools for computer vision, including face detection and recognition.

* Project Implementation: The system will be implemented in Python. The following steps will be involved in the implementation:

      1. Collect a dataset of face images. The dataset should include images of different people from different angles and lighting conditions.

      2. Train a face recognition model. The face recognition model will be trained on the collected dataset of face images.

      3. Implement the face detection and recognition algorithm. The face detection and recognition algorithm will be implemented using the Dlib library.
## Demo/ Screenshots
# TO BE ADDED
### About Accuracy
   - When using a distance threshold of `0.6`, the dlib model obtains an accuracy of `99.38%` on the standard LFW face recognition benchmark.

### About algorithm
   - Residual Neural Network/ CNN
   - This model is a ResNet network with 29 conv layers. 
   
It's essentially a version of the ResNet-34 network from the paper Deep Residual Learning for Image Recognition by He, Zhang, Ren, and Sun with a few layers removed and the number of filters per layer reduced by half.



## Steps 

#### Download Required Files: [--LINK HERE--](https://drive.google.com/drive/folders/1iUx3uh9c9DTnBhEiNltXpYK1uhgEiim6?usp=share_link)
- dlib_face_recognition_resnet_model_v1.dat
- shape_predictor_68_face_landmarks.dat

- #### Save these files into this path `data\data_dlib`
- #### Verify file path into `constants.py`
```
https://drive.google.com/drive/folders/1iUx3uh9c9DTnBhEiNltXpYK1uhgEiim6?usp=share_link
```

1. Git clone source code
```bash
  git clone https://github.com/coneypo/Dlib_face_recognition_from_camera
```

2. Install some python packages needed
```bash
  pip install -r requirements.txt
```

3. Register faces with Tkinter GUI
```bash
  # Install Tkinter
  sudo apt-get install python3-tk python3-pil python3-pil.imagetk

  python3 face_register.py
```

4. Features extraction and save into `features_all.csv`

```bash
  python3 face_train.py
```

5. Real-time face recognition
```bash
  python3 face_recognition.py
```

## About Source Code
- Code structure:

```
.
├── get_faces_from_camera.py                        # Step 1. Face register GUI with OpenCV
├── get_faces_from_camera_tkinter.py                # Step 1. Face register GUI with Tkinter
├── features_extraction_to_csv.py                   # Step 2. Feature extraction
├── face_reco_from_camera.py                        # Step 3. Face recognizer
├── face_reco_from_camera_single_face.py            # Step 3. Face recognizer for single person
├── face_reco_from_camera_ot.py                     # Step 3. Face recognizer with OT
├── face_descriptor_from_camera.py                  # Face descriptor computation
├── how_to_use_camera.py                            # Use the default camera by opencv
├── data
│   ├── data_dlib                                   # Dlib's model
│   │   ├── dlib_face_recognition_resnet_model_v1.dat
│   │   └── shape_predictor_68_face_landmarks.dat
│   ├── data_faces_from_camera                      # Face images captured from camera (will generate after step 1)
│   │   ├── person_1
│   │   │   ├── img_face_1.jpg
│   │   │   └── img_face_2.jpg
│   │   └── person_2
│   │       └── img_face_1.jpg
│   │       └── img_face_2.jpg
│   └── features_all.csv                            # CSV to save all the features of known faces (will generate after step 2)
├── README.rst
└── requirements.txt                                # Some python packages needed
```
# Dlib related functions used in this repo:

1. Dlib (based on HOG), output: `<class 'dlib.dlib.rectangles'>` / Dlib frontal face detector.

```
detector = dlib.get_frontal_face_detector()
faces = detector(img_gray, 0)
```

2. Dlib landmark, output: `<class 'dlib.dlib.full_object_detection'>` / Dlib face landmark predictor, will use shape_predictor_68_face_landmarks.dat

```
# This is trained on the ibug 300-W dataset (https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
# Also note that this model file is designed for use with dlib's HOG face detector.
# That is, it expects the bounding boxes from the face detector to be aligned a certain way,
the way dlib's HOG face detector does it.
# It won't work as well when used with a face detector that produces differently aligned boxes,
# such as the CNN based mmod_human_face_detector.dat face detector.

predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
shape = predictor(img_rd, faces[i])
```
3. Face recognition model, the object maps human faces into 128D vectors

```
face_rec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

```
## Source Code
1. `face_register.py`

Face information collection and entry / Face register with OpenCV GUI

- Please note that when storing face pictures, the rectangular frame should not exceed the range of the camera, otherwise it cannot be saved locally.

- There will be an "out of range" reminder if it exceeds;

2. `face_train.py`
From the image file saved in the previous step, extract the face data and save it in CSV / Extract features from face images saved.

  - Will generate a store of all feature face data `features_all.csv`
  - Size: n*129 , n means n faces you registered and 129 means face name + 128D features of this face.

3. `face_recognition.py`

This step will call the camera for real-time face recognition; / This part will implement real-time face recognition;

  - Compare the captured face data with the previously stored face data to calculate the Euclidean distance, so as to determine whether they are the same person.
  - Compare the faces captured from camera with the faces you have registered which are saved in features_all.csv.
## More

  - Blog: https://www.cnblogs.com/AdaminXie/p/9010298.html
  - The update on the OT part is in Blog: https://www.cnblogs.com/AdaminXie/p/13566269.html