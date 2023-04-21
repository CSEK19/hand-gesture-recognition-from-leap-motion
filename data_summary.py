import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("pose_data_2004.pkl", "rb") as f:
  data = pickle.load(f)


classes = ['palm', 'left', 'right',  'up', 'down', 'palm_l', 'palm_r', 'palm_u', 'fist', 'hook', 'stop', 'thumb_in', 'negative']
Xs, ys = data

# Start plotting histogram
_, counts = np.unique(ys, return_counts=True)
print(counts)

fig, ax = plt.subplots(1, 1, figsize=(10,7))
ax.bar(classes[:-1], counts, align='center')
ax.set_title("Pose dataset distribution")
ax.set_xlabel("Pose name")
ax.set_ylabel("Number of samples")

plt.show()
