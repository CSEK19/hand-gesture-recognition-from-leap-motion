import numpy as np
import os, json, glob
import pickle
import matplotlib.pyplot as plt
import random
import math
import cv2

gestures = ['d_fist', 'd_left', 'd_right', 'd_up', 'd_down', 'd_rotate_left', 'd_rotate_right', 'd_stop', 'd_thumb', 'd_negative']
gesture_map = {
    'd_fist' : 0,
    'd_left' : 1,
    'd_right': 2,
    'd_up': 3,
    'd_down': 4,
    'd_rotate_left': 5,
    'd_rotate_right': 6,
    'd_stop': 7,
    'd_thumb': 8,
    'd_negative': 9,
}

data_folder = 'data_collection/LeapDatav4'

gesture_data = []

for samples_path in glob.glob(f'{data_folder}\\*\\*'):
    # extract path 
    _, subject, label = samples_path.split('\\')

    if label in gesture_map and label not in ['d_rotate_left', 'd_rotate_right']:
        for json_path in glob.glob(f"{samples_path}\\*\\*.json"):
            path_traceback = json_path.rsplit('\\', 1)[0]
            # load json file
            with open(json_path) as json_file:
                data = json.load(json_file)

            gesture_idx = gesture_map[label]
            history_angles = []
            for x in data:
                if len(data[x]["hands"]) == 1:
                    palm = data[x]["hands"][0]["palm"]
                    thumb = data[x]["hands"][0]["thumb"]
                    index = data[x]["hands"][0]["index"]
                    middle = data[x]["hands"][0]["middle"]
                    ring = data[x]["hands"][0]["ring"]
                    pinky = data[x]["hands"][0]["pinky"]
                    yaw = data[x]["hands"][0]["yaw"]
                    pitch = data[x]["hands"][0]["pitch"]
                    roll = data[x]["hands"][0]["roll"]
                    handedness = 0 if data[x]["hands"][0]["handedness"] == 'Left' else 1
                    raw_data_lst = palm + thumb + index + middle + ring + pinky + [yaw, pitch, roll, handedness]
                    history_angles.append(roll * 180/math.pi)
                else:
                    pass
            
            plt.plot(history_angles)
            plt.title(path_traceback)
            plt.show()

