import numpy as np
import pickle 

class StaticHandPoseClassifier:
  def __init__(self, weight_file, std_scale):
    self.weight = pickle.load(open(weight_file, 'rb'))
    self.std_scale = pickle.load(open(std_scale, 'rb'))

    self.classes = ['palm', 'fist', 'stop', 'thumb_in']

  def feature_extract_11_04(self, X):

    def feature_extract_element(x):
      a = x[0:66].reshape(22,3)
      a = a - a[0]
      a = a[[0,5,9,13,17,21]]
      pdist_a = pdist(a, metric='euclidean').tolist() 
      return np.array(pdist_a)
    return np.array([feature_extract_element(x) for x in X])
  
  def predict(self, X): #shape: (-1, 70)
    #preprocess
    X = self.feature_extract_11_04(X)
    X = self.std_scale.transform(X)

    y = self.weight.predict(X)
    return [self.classes[i] for i in y]
  
  def predict_proba(self, X):
    try:
      X = self.feature_extract_11_04(X)
      X = self.std_scale.transform(X)
      return self.weight.predict_proba(X)
    except:
      return None