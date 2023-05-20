# ===== IMPORTS =====
import cv2
import threading
import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import threading
import _thread as thread
import cv2
import urllib.request
import numpy as np
import time
import constants

# ===== INITIALIZATION 1 =====
detector = dlib.get_frontal_face_detector()

# Get face landmarks predictor
predictor = constants.predictor_file_path
# predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = constants.face_reco_model_file_path
# face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

'''
Download Links for both files mentioned above: 
https://drive.google.com/drive/folders/1iUx3uh9c9DTnBhEiNltXpYK1uhgEiim6?usp=share_link
'''

# ===== INITIALIZATION 2 =====
font = cv2.FONT_ITALIC
frame_time = 0
frame_start_time = 0
fps = 0
fps_show = 0
start_time = time.time()

# count for frame
frame_cnt = 0

# Save the features of faces in the database
face_features_known_list = []
# Save the name of faces in the database
face_name_known_list = []

# List to save centroid positions of ROI in frame N-1 and N
last_frame_face_centroid_list = []
current_frame_face_centroid_list = []

# List to save names of objects in frame N-1 and N
last_frame_face_name_list = []
current_frame_face_name_list = []

# cnt for faces in frame N-1 and N
last_frame_face_cnt = 0
current_frame_face_cnt = 0

# Save the e-distance for faceX when recognizing
current_frame_face_X_e_distance_list = []

# Save the positions and names of current faces captured
current_frame_face_position_list = []
# Save the features of people in current frame
current_frame_face_feature_list = []

# e distance between centroid of ROI in last and current frame
last_current_frame_centroid_e_distance = 0

#  Reclassify after 'reclassify_interval' frames
#  "unknown" ,  reclassify_interval_cnt  reclassify_interval ,
reclassify_interval_cnt = 0
reclassify_interval = 10


# ===== MAIN =====

class CamThread(threading.Thread):
    def __init__(self, preview_name, cam_id):
        threading.Thread.__init__(self)
        self.previewName = preview_name
        self.camID = cam_id

    def run(self):
        print("Starting " + self.previewName)
        # camPreview(self.previewName, self.camID)


# ===== THREADING =====

for each_camera in constants.cameras:
    #print(each_camera) # Camera Titles
    #print(constants.cameras[each_camera]) # Camera Sources
    thread = CamThread(each_camera, constants.cameras[each_camera])
    thread.start()
