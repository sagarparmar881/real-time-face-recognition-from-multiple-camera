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
predictor = dlib.shape_predictor(constants.predictor_file_path)
# predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1(constants.face_reco_model_file_path)
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
        cam_preview(self.previewName, self.camID)


def update_fps():
    now = time.time()
    global start_time, fps, frame_start_time, fps_show, frame_time
    #  Refresh fps per second
    if str(start_time).split(".")[0] != str(now).split(".")[0]:
        fps_show = fps
    start_time = now
    frame_time = now - frame_start_time
    fps = 1.0 / frame_time
    frame_start_time = now


def get_face_database():
    if os.path.exists(constants.features_csv_file_path):
        path_features_known_csv = constants.features_csv_file_path
        csv_rd = pd.read_csv(path_features_known_csv, header=None)
        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            face_name_known_list.append(csv_rd.iloc[i][0])
            for j in range(1, 129):
                if csv_rd.iloc[i][j] == "":
                    features_someone_arr.append("0")
                else:
                    features_someone_arr.append(csv_rd.iloc[i][j])
            face_features_known_list.append(features_someone_arr)
        logging.info("Faces in Database %d", len(face_features_known_list))
        return 1
    else:
        logging.warning("'features_all.csv' not found!")
        logging.warning(
            "Please run 'get_faces_from_camera.py' "
            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'"
        )
        return 0


def centroid_tracker():
    for i in range(len(current_frame_face_centroid_list)):
        e_distance_current_frame_person_x_list = []
        # For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
        for j in range(len(last_frame_face_centroid_list)):
            last_current_frame_centroid_e_distance = return_euclidean_distance(
                current_frame_face_centroid_list[i], last_frame_face_centroid_list[j]
            )

            e_distance_current_frame_person_x_list.append(
                last_current_frame_centroid_e_distance
            )

        last_frame_num = e_distance_current_frame_person_x_list.index(
            min(e_distance_current_frame_person_x_list)
        )
        current_frame_face_name_list[i] = last_frame_face_name_list[last_frame_num]


def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


def draw_note(img_rd):
    global font
    cv2.putText(
        img_rd,
        "Frame: " + str(frame_cnt),
        (10, 20),
        font,
        0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_rd,
        "FPS: " + str(fps_show.__round__(2)),
        (150, 20),
        font,
        0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_rd,
        "Faces: " + str(current_frame_face_cnt),
        (300, 20),
        font,
        0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_rd, "Q: Quit", (10, 450), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )

    for i in range(len(current_frame_face_name_list)):
        pass
        # img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
        #     [int(current_frame_face_centroid_list[i][0]), int(current_frame_face_centroid_list[i][1])]),
        #                      font,
        #                      0.8, (255, 190, 0),
        #                      1,
        #                      cv2.LINE_AA)


def cam_preview(preview_name, cam_id):
    cv2.namedWindow(preview_name)
    cam = cv2.VideoCapture(cam_id)
    # font = cv2.FONT_ITALIC

    if cam.isOpened():
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        rval, frame = cam.read()
        # ==== START =====
        if get_face_database():
            while cam.isOpened():
                global frame_cnt, last_frame_face_cnt, current_frame_face_cnt, last_frame_face_name_list
                global last_frame_face_centroid_list, current_frame_face_centroid_list, current_frame_face_position_list
                global last_frame_face_name_list, current_frame_face_name_list, reclassify_interval_cnt

                frame_cnt += 1
                logging.debug("Frame " + str(frame_cnt) + " starts")
                flag, img_rd = cam.read()
                keyboard_key = cv2.waitKey(1)

                # Detect faces for frame X
                faces = detector(img_rd, 0)

                # Update count for faces in frames
                last_frame_face_cnt = current_frame_face_cnt
                current_frame_face_cnt = len(faces)

                # Update the face name list in last frame
                last_frame_face_name_list = current_frame_face_name_list[:]

                # Update frame centroid list
                last_frame_face_centroid_list = current_frame_face_centroid_list
                current_frame_face_centroid_list = []

                # If count not changes
                if (current_frame_face_cnt == last_frame_face_cnt) and (
                        reclassify_interval_cnt != reclassify_interval):
                    logging.debug("Scene 1:  / No face count changes in this frame!")
                    current_frame_face_position_list = []

                    if "unknown" in current_frame_face_name_list:
                        logging.debug("  ,  reclassify_interval_cnt ")
                        reclassify_interval_cnt += 1

                    if current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                    # Multi-faces in current frame, use centroid-tracker to track
                    if current_frame_face_cnt != 1:
                        centroid_tracker()

                    for i in range(current_frame_face_cnt):
                        # Write names under ROI
                        img_rd = cv2.putText(img_rd, current_frame_face_name_list[i],
                                             current_frame_face_position_list[i], font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    draw_note(img_rd)

                # If count of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("Scene 2: Faces cnt changes in this frame")
                    current_frame_face_position_list = []
                    # current_frame_face_feature_list = []
                    reclassify_interval_cnt = 0

                    # Face count decreases: 1->0, 2->1, ...
                    if current_frame_face_cnt == 0:
                        logging.debug("Scene 2.1: No faces in this frame!!!")
                        # clear list of names and features
                        current_frame_face_name_list = []
                    # Face count increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug("Scene 2.2: Get faces in this frame and do face recognition")
                        current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            current_frame_face_name_list.append("unknown")

                        # Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("  For face %d in current frame:", k + 1)
                            current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            # current_frame_face_X_e_distance_list = []

                            # Positions of faces captured
                            current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # For every faces detected, compare the faces in the database
                            for i in range(len(face_features_known_list)):
                                #  q
                                if str(face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = return_euclidean_distance(
                                        current_frame_face_feature_list[k],
                                        face_features_known_list[i])
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    #  person_X
                                    current_frame_face_X_e_distance_list.append(999999999)

                            # Find the one with minimum e distance
                            similar_person_num = current_frame_face_X_e_distance_list.index(
                                min(current_frame_face_X_e_distance_list))

                            if min(current_frame_face_X_e_distance_list) < 0.4:
                                current_frame_face_name_list[k] = face_name_known_list[similar_person_num]
                                logging.debug("  Face recognition result: %s",
                                              face_name_known_list[similar_person_num])
                            else:
                                logging.debug("  Face recognition result: Unknown person")

                        # Add note on cv2 window
                        draw_note(img_rd)

                        # cv2.imwrite("debug/debug_" + str(frame_cnt) + ".png", img_rd) # Dump current frame image if needed

                # Press 'q' to exit
                if keyboard_key == ord('q'):
                    break

                update_fps()
                cv2.imshow(preview_name, img_rd)

                logging.debug("Frame ends.")
        # ==== END =====
        if keyboard_key == 113:  # exit on "Q"
            break
    cv2.destroyWindow(preview_name)


# ===== THREADING =====

cameras = {
    "camera_feed_0": 0,
    "camera_feed_1": "sample_videos/sample_video_01.mp4",
}
for each_camera in cameras:
    # print(each_camera) # Camera Titles
    # print(constants.cameras[each_camera]) # Camera Sources
    thread = CamThread(str(each_camera), cameras[each_camera])
    thread.start()
