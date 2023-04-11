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


fast_scandir("LeapData")
# print(lst_json_file)
lst_gesture = {
    'down' : 0,
    'fist' : 1,
    'left': 2,
    'palm': 3,
    'right': 4,
    'rotate': 5,
    'up': 6,
    'negative': 7,

}

lst_d_gesture = {
    'd_down' : 0,
    'd_fist' : 1,
    'd_left': 2,
    'd_right': 3,
    'd_rotate': 4,
    'd_up': 5,
    'd_negative': 6,

}

dynamic_prefix = "_"
arr_data = []
arr_name = []
arr_data_dynamic = []

print(lst_json_file[0])
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
                    skeleton = data[x]["hands"][0]["skeleton"] 
                    yaw = data[x]["hands"][0]["yaw"]
                    pitch = data[x]["hands"][0]["pitch"]
                    roll = data[x]["hands"][0]["roll"]
                    handedness = 0 if data[x]["hands"][0]["handedness"] == 'left' else 1
                    raw_data_lst = skeleton + [yaw] + [pitch] + [roll] + [handedness]
                    arr_tmp.append(raw_data_lst)
                else:
                    arr_tmp.append([])

            arr_data_dynamic.append([arr_tmp, d_gesture, path_traceback])
        else:
            getsture = lst_gesture[third_last_dir]
            for x in data:  
                if(len(data[x]["hands"])):
                    skeleton = data[x]["hands"][0]["skeleton"] 
                    yaw = data[x]["hands"][0]["yaw"]
                    pitch = data[x]["hands"][0]["pitch"]
                    roll = data[x]["hands"][0]["roll"]
                    handedness = 0 if data[x]["hands"][0]["handedness"] == 'left' else 1
                    raw_data_lst = skeleton + [yaw] + [pitch] + [roll] + [handedness]
                    arr_data.append(raw_data_lst)
                    arr_name.append(getsture)

array_data_x = np.array(arr_data)
array_data_y = np.array(arr_name)

with open("array_x.pkl", "wb") as f:
  pickle.dump(array_data_x, f)
with open("array_y.pkl", "wb") as f:
  pickle.dump(array_data_y, f)
with open("array_dynamic.pkl", "wb") as f:
  pickle.dump(arr_data_dynamic, f)