import numpy as np
import os, json, glob
import pickle
import matplotlib.pyplot as plt
import random

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

data_folder = 'data_collection/LeapDatav4'
train_subjects = ['Khang', 'Phat', 'Sang', 'Xuan']
test_subjects = ['PhatLe', 'Mom']

pose_data_train = []
pose_data_test = []
gesture_data = []


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
                    handedness = 0 if data[x]["hands"][0]["handedness"] == 'left' else 1
                    raw_data_lst = palm + thumb + index + middle + ring + pinky + [yaw, pitch, roll, handedness]
                    
                    samples.append([raw_data_lst, pose_idx])

        # sample the data to avoid imbalance
        if len(samples) > 60:
            samples = random.sample(samples, 60)
        if subject in train_subjects:
            pose_data_train += samples
        elif subject in test_subjects:
            pose_data_test += samples
    
    # process gesture data
    elif label in gestures:
        for json_path in glob.glob(f"{samples_path}\\*\\*.json"):
            path_traceback = json_path.rsplit('\\', 1)[0] + '\\video.mp4'
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
                    handedness = 0 if data[x]["hands"][0]["handedness"] == 'left' else 1
                    raw_data_lst = palm + thumb + index + middle + ring + pinky + [yaw, pitch, roll, handedness]
                    samples.append(raw_data_lst)
                else:
                    samples.append([])

            gesture_data.append([samples, gesture_idx, path_traceback])


with open("data_collection/array_data_2504/pose_data_train_2504.pkl", "wb") as f:
  pickle.dump(pose_data_train, f)

with open("data_collection/array_data_2504/pose_data_test_2504.pkl", "wb") as f:
  pickle.dump(pose_data_test, f)

with open("data_collection/array_data_2504/gesture_data_2504.pkl", "wb") as f:
  pickle.dump(gesture_data, f)

# Histogram for pose training set
_, pose_counts = np.unique([x[1] for x in pose_data_train], return_counts=True)
print(pose_counts)

fig, ax = plt.subplots(1, 1, figsize=(10,7))
ax.bar(poses, pose_counts, align='center')
ax.set_title("Pose training set distribution")
ax.set_xlabel("Pose name")
ax.set_ylabel("Number of samples")
plt.show()

# Histogram for pose test set
_, pose_counts = np.unique([x[1] for x in pose_data_test], return_counts=True)
print(pose_counts)

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
ax.bar(gestures, gesture_counts, align='center')
ax.set_title("Gesture dataset distribution")
ax.set_xlabel("Gesture name")
ax.set_ylabel("Number of samples")
plt.show()
