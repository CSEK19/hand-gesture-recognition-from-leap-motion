import numpy as np
import pickle 
from scipy.spatial.distance import pdist

def angle(v1, v2):
  unit_v1 = v1 / np.linalg.norm(v1)
  unit_v2 = v2 / np.linalg.norm(v2)
  dot_product = np.dot(unit_v1, unit_v2)
  angle = np.arccos(dot_product)
  return angle

class StaticHandPoseClassifier:
  def __init__(self, model_weight, scaler_weight=None):
    self.model = pickle.load(open(model_weight, 'rb'))
    self.scaler = pickle.load(open(scaler_weight, 'rb'))
    self.classes = ['palm', 'left', 'right',  'up', 'down', 'palm_l', 'palm_r', 'palm_u', \
      'fist', 'hook', 'stop', 'thumb_in', 'negative']

  def feature_extraction(self, X):
    X = np.array(X)
    yaw, pitch, roll, handedness = X[-4:]
    joints = X[:-4].reshape(-1,3)
    # print(joints.shape)
    palm = joints[0]
    # angle between thumb joints
    thumb = joints[1:6]
    thumb_bones = [thumb[i+1]-thumb[i] for i in range(2,4)]
    thumb_angles = np.array([angle(thumb_bones[i], thumb_bones[i+1]) for i in range(0,1)])
    # angle between index joints
    index = joints[6:11]
    index_bones = [index[i+1]-index[i] for i in range(1,4)]
    index_angles = np.array([angle(index_bones[i], index_bones[i+1]) for i in range(0,2)])
    # angle between middle joints
    middle = joints[11:16]
    middle_bones = [middle[i+1]-middle[i] for i in range(1,4)]
    middle_angles = np.array([angle(middle_bones[i], middle_bones[i+1]) for i in range(0,2)])
    # angle between ring joints
    ring = joints[16:21]
    ring_bones = [ring[i+1]-ring[i] for i in range(1,4)]
    ring_angles = np.array([angle(ring_bones[i], ring_bones[i+1]) for i in range(0,2)])
    # angle between pinky joints
    pinky = joints[21:26]
    pinky_bones = [pinky[i+1]-pinky[i] for i in range(1,4)]
    pinky_angles = np.array([angle(pinky_bones[i], pinky_bones[i+1]) for i in range(0,2)])


    fingertip_joints = np.array([thumb[-1], index[-1], middle[-1], ring[-1], pinky[-1]])
    fingertip_vectors = fingertip_joints - palm
    fingertip_angles = np.array([angle(fingertip_vectors[i], fingertip_vectors[i+1]) for i in range(0,4)])
    feature_vector = np.concatenate((fingertip_angles, thumb_angles, index_angles, middle_angles, ring_angles, pinky_angles, np.array([yaw, pitch, roll])), axis=0)
    # feature_vector = np.concatenate((fingertip_angles, np.array([yaw, pitch, roll])), axis=0)

    # calculate distances
    if self.scaler is not None:
      pairwise_distances = pdist(np.concatenate((palm.reshape(1,-1), fingertip_joints), axis=0), metric='euclidean')
      scaled_distances = self.scaler.transform(pairwise_distances.reshape(1,-1))[0]
      feature_vector = np.concatenate((feature_vector, scaled_distances))

    return feature_vector
  
  def predict(self, X): #shape: (-1, 70)
    #preprocess
    X = self.feature_extraction(X)
    y = self.model.predict(X.reshape(1,-1))[0]
    # return the class name
    return self.classes[y]
  
  def predict_proba(self, X):
    # print(self.feature_extraction(X).shape)
    # try:
      #preprocess
    X = self.feature_extraction(X)
    y = self.model.predict_proba(X.reshape(1,-1))[0]
    # return the class name and score
    return self.classes[np.argmax(y)], np.max(y)
    # except:
    #   return None


# https://www.webucator.com/article/python-clocks-explained/
# https://sci-hub.se/https://ieeexplore.ieee.org/document/8572153