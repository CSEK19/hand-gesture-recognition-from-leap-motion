import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("data_collection/pose_data_palm_fist_2204_test.pkl", "rb") as f:
  data = pickle.load(f)


# classes = ['palm', 'left', 'right',  'up', 'down', 'palm_l', 'palm_r', 'palm_u', 'fist', 'hook', 'stop', 'thumb_in', 'negative']
classes = ['palm', 'fist']
Xs, ys = data

# Start plotting histogram
_, counts = np.unique(ys, return_counts=True)
print(counts)

fig, ax = plt.subplots(1, 1, figsize=(10,7))
ax.bar(classes[:], counts, align='center')
ax.set_title("Pose dataset distribution")
ax.set_xlabel("Pose name")
ax.set_ylabel("Number of samples")

plt.show()
