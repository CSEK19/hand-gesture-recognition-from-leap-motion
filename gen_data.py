import numpy as np
import os, json, glob
import pickle
import matplotlib.pyplot as plt
import random
import math
import cv2

poses = ['palm', 'left', 'right',  'up', 'down', 'palm_left', 'palm_right', 'palm_up', 'fist', 'hook', 'stop', 'thumb_in']
gestures = ['d_fist', 'd_left', 'd_right', 'd_up', 'd_down', 'd_rotate_left', 'd_rotate_right', 'd_stop', 'd_thumb', 'd_negative']
pose_map = {
    'palm' : 0,
    'left': 1,
    'right': 2,
    'up': 3,
    'down': 4,
    'palm_left': 5,
    'palm_right': 6,
    'palm_up': 7,
    'fist': 8,
    'hook': 9,
    'stop': 10,
    'thumb_in': 11,
    'negative': 12
}
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

data_folder = 'LeapData/data_collection/LeapDatav5'
train_subjects = ['Khang', 'Phat', 'Sang', 'Xuan']
test_subjects = ['PhatLe', 'Mom']

pose_data_train = []
pose_data_test = []
gesture_data = []

# gesture video lengths
gesture_data_lengths = {}

for samples_path in glob.glob(f'{data_folder}\\*\\*'):
    # extract path 
    _, subject, label = samples_path.split('\\')

    # process pose data for each class performed by each user
    if label in poses:
        samples = []
        for json_path in glob.glob(f"{samples_path}\\*\\*.json"):
            path_traceback = json_path.rsplit('\\', 1)[0] + '\\video.mp4'
            # load json file
            with open(json_path) as json_file:
                data = json.load(json_file)
            pose_idx = pose_map[label]
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
                    
                    if label == 'palm_up' and handedness == 1 and roll > 0:
                        roll = -2 * math.pi + roll
                    elif label == 'palm_up' and handedness == 0 and roll < 0:
                        roll = 2 * math.pi  + roll

                    raw_data_lst = palm + thumb + index + middle + ring + pinky + [yaw, pitch, roll, handedness]
                    # Debug
                    pitch_degree = pitch * 180 / math.pi
                    roll_degree = roll * 180 / math.pi
                    yaw_degree = yaw * 180 / math.pi
                    # if label == 'up' and pitch_degree < 15.0:
                    #     print(json_path, x, pitch_degree)
                    # if label == 'left' and yaw_degree > -15:
                    #     print(json_path, x, yaw_degree)

                    # if label == 'palm' and yaw_degree < -10:
                    #     print(json_path, x, yaw_degree)

                    # if label == 'palm_up' and abs(roll_degree) < 140:
                    #     print(json_path, x, roll_degree)

                    # if label == 'palm_up' and handedness == 1 and abs(roll_degree) < 120:
                    #     print(json_path, x, roll_degree)

                    # if label == 'hook' and (pitch_degree > 15 or pitch_degree < -15):
                    #     print(json_path, x, pitch_degree)

                    # if label == 'down' and (pitch_degree > -20):
                    #     print(json_path, x, pitch_degree)

                    # if label == 'down' and (pitch_degree > -25 ):
                    #     print(json_path, x, pitch_degree)

                    
                    samples.append([raw_data_lst, pose_idx, f'{json_path}_{x}'])


        # sample the data to avoid imbalance
        # if len(samples) > 60:
        #     samples = random.sample(samples, 60)
        if subject in train_subjects:
            pose_data_train += samples
        elif subject in test_subjects:
            pose_data_test += samples
    
    # process gesture data
    elif label in gestures:
        for json_path in glob.glob(f"{samples_path}\\*\\*.json"):
            path_traceback = json_path.rsplit('\\', 1)[0]
            # load json file
            with open(json_path) as json_file:
                data = json.load(json_file)

            gesture_idx = gesture_map[label]
            samples = []
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
                    samples.append(raw_data_lst)
                else:
                    samples.append([])

            gesture_data.append([samples, gesture_idx, path_traceback])
        
        # summary of video length
        if label not in gesture_data_lengths:
            gesture_data_lengths[label] = []
        for video_path in glob.glob(f"{samples_path}\\*\\*.mp4"):
            cap = cv2.VideoCapture(video_path)
            # fps = cap.get(cv2.CAP_PROP_FPS)
            fps = 60
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count/fps
            gesture_data_lengths[label].append(duration)


# summary of video length
all_gesture_lengths = []
for gesture in gesture_data_lengths:
    lengths = gesture_data_lengths[gesture]
    all_gesture_lengths += lengths
    arr_lengths = np.array(lengths)
    mean = np.mean(arr_lengths)
    variance = np.var(arr_lengths)
    std = np.std(arr_lengths)
    median = np.median(arr_lengths)
    min_l = np.min(arr_lengths)
    max_l = np.max(arr_lengths)
    print(f"Gesture {gesture}: mean = {mean}, std = {std}, median = {median}, min = {min_l}, max = {max_l}")

all_gesture_lengths = np.array(all_gesture_lengths)
mean = np.mean(all_gesture_lengths)
variance = np.var(all_gesture_lengths)
std = np.std(all_gesture_lengths)
median = np.median(all_gesture_lengths)
min_l = np.min(all_gesture_lengths)
max_l = np.max(all_gesture_lengths)
print(f"All gestures: mean={mean}, std={std}, median={median}, min={min_l}, max={max_l}")

# with open("data_collection/array_data/pose_data_train_v5.pkl", "wb") as f:
#   pickle.dump(pose_data_train, f)

# with open("data_collection/array_data/pose_data_test_v5.pkl", "wb") as f:
#   pickle.dump(pose_data_test, f)

# with open("data_collection/array_data/gesture_data_v5.pkl", "wb") as f:
#   pickle.dump(gesture_data, f)

# Histogram for pose training set
_, pose_counts = np.unique([x[1] for x in pose_data_train], return_counts=True)
print(pose_counts, sum(pose_counts))

fig, ax = plt.subplots(1, 1, figsize=(10,7))
ax.bar(poses, pose_counts, align='center')
ax.set_title("Pose training set distribution")
ax.set_xlabel("Pose name")
ax.set_ylabel("Number of samples")
plt.show()

# Histogram for pose test set
_, pose_counts = np.unique([x[1] for x in pose_data_test], return_counts=True)
print(pose_counts, sum(pose_counts))

fig, ax = plt.subplots(1, 1, figsize=(10,7))
ax.bar(poses, pose_counts, align='center')
ax.set_title("Pose test set distribution")
ax.set_xlabel("Pose name")
ax.set_ylabel("Number of samples")
plt.show()

# Histogram for gesture data
_, gesture_counts = np.unique([x[1] for x in gesture_data], return_counts=True)
print(gesture_counts)

fig, ax = plt.subplots(1, 1, figsize=(10,7))
gesture_display = ['close\nfist','move\nleft','move\nright','move\nup','move\ndown','rotate\nleft','rotate\nright','stop','thumb in', 'negative']
ax.bar(gesture_display, gesture_counts, align='center')
ax.set_title("Gesture dataset distribution")
ax.set_xlabel("Gesture name")
ax.set_ylabel("Number of samples")
plt.show()
