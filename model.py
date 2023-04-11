import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist

class StaticHandPoseClassifier:
  def __init__(self, weight_file, std_scale):
    self.weight = pickle.load(open(weight_file, 'rb'))
    self.std_scale = pickle.load(open(std_scale, 'rb'))

    self.classes = ['palm', 'fist', 'stop', 'left', 'right', 'up', 'down', 'rotate', 'thumb_in']

  def feature_extract(self, X):
    def angle_v1v2(v1, v2):
      def unit_vector(vector):
        return vector / np.linalg.norm(vector)
      v1_u = unit_vector(v1)
      v2_u = unit_vector(v2)
      return abs(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    def feature_extract_element(x):
      a = x[0:66].reshape(22,3)
      a = a - a[0]
      a = a[[0,5,9,13,17,21]]
      pdist_a = pdist(a, metric='euclidean').tolist() 
      angle_a = [angle_v1v2(a[1], a[2]),angle_v1v2(a[2], a[3]),angle_v1v2(a[3], a[4]), angle_v1v2(a[4], a[5])]
      y_p_r = x[66:-1].tolist() 

      return np.array(pdist_a+angle_a+y_p_r)
    return np.array([feature_extract_element(x) for x in X])
  
  def predict(self, X): #shape: (-1, 70)
    #preprocess
    X = self.feature_extract(X)
    X = self.std_scale.transform(X)

    y = self.weight.predict(X)
    # return [self.classes[i] for i in y]
    return y
  
  def predict_proba(self, X):
    try:
    #preprocess
      X = self.feature_extract(X)
      X = self.std_scale.transform(X)
      return self.weight.predict_proba(X)
    except:
      return None