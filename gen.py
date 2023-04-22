import numpy as np
import os, json
import pickle

lst_json_file = []

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    if not subfolders:
        for file in os.listdir(dirname):
            if file.endswith('.json'):
                lst_json_file.append(f"{dirname}\\{file}")
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders


fast_scandir("data_collection/LeapDatav4_test")
# print(lst_json_file)
# lst_gesture = {
#     'palm' : 0,
#     'left': 1,
#     'right': 2,
#     'up': 3,
#     'down': 4,
#     'palm_left': 5,
#     'palm_right': 6,
#     'palm_up': 7,
#     'fist': 8,
#     'hook': 9,
#     'stop': 10,
#     'thumb_in': 11,
#     'negative': 12
# }

lst_gesture = {
    'palm': 0,
    'fist': 1
}

lst_d_gesture = {
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

dynamic_prefix = "d_"
arr_data = []
arr_name = []
arr_data_dynamic = []

# print(lst_json_file[0])
for json_file in lst_json_file:
    path = json_file

    dir_list = path.split('\\')
    path_traceback = path.rsplit('\\', 1)[0] + '\\video.mp4'
    third_last_dir = dir_list[-3]
    with open(path) as json_file:
        data = json.load(json_file)

        # For dynamic gesture
        if dynamic_prefix in third_last_dir:
            d_gesture= lst_d_gesture[third_last_dir]
            arr_tmp = []
            for x in data:  
                if(len(data[x]["hands"])):
                    if len(data[x]["hands"]) > 1:
                        print(path)
                        print(data[x]["hands"][0]["palm"])
                        print(data[x]["hands"][1]["palm"])
                        continue
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
                    arr_tmp.append(raw_data_lst)
                else:
                    arr_tmp.append([])

            arr_data_dynamic.append([arr_tmp, d_gesture, path_traceback])
        else:
            if third_last_dir not in lst_gesture:
                continue
            getsture = lst_gesture[third_last_dir]
            for x in data:
                if(len(data[x]["hands"])):
                    # print('hehe')
                    assert len(data[x]["hands"]) == 1
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
                    arr_data.append(raw_data_lst)
                    arr_name.append(getsture)

array_data_x = np.array(arr_data)
array_data_y = np.array(arr_name)

with open("data_collection/pose_data_palm_fist_2204_test.pkl", "wb") as f:
  pickle.dump([array_data_x, array_data_y], f)

# with open("gesture_data_2204.pkl", "wb") as f:
#   pickle.dump(arr_data_dynamic, f)