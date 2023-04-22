import numpy as np
import math

POSES = ['palm', 'fist', 'stop', 'left', 'right', 'up', 'down', 'rotate', 'thumb_in', 'negative']
GESTURES = ['close fist', 'move left', 'move right', 'move up', 'move down', 'rotate', 'stop', 'thumb_in', 'negative']
START_END_MATCHES_LABEL = {
  'move down': [['palm', 'down']],
  'move up': [['palm','up']],
  'move left': [['palm','left']],
  'move right': [['palm', 'right']],
  'close fist': [['palm', 'fist']],
  'rotate right': [['palm', 'palm_l']],
  'rotate left': [['palm', 'palm_r']]
  }

# START_END_MATCHES = {}
# for gesture in START_END_MATCHES_LABEL:
#   gesture_id = GESTURES.index(gesture)
#   START_END_MATCHES[gesture_id] = []
#   [START_END_MATCHES[gesture_id].append(POSES.index(x)) for x in START_END_MATCHES_LABEL[gesture]]

# print(START_END_MATCHES)

class HandGestureRecognizer:
  def __init__(self, pose_classifier):
    self.classifier = pose_classifier
    self.buffer = []

  def detect(self, feature):
    # Make detection
    pose = self.detect_pose(feature)
    
    # detect dynamic gesture if the static pose is not negative
    if pose != "negative":
      gesture = self.detect_gesture(pose)
      # print result if key poses matched
      if gesture != "negative":
        print(gesture)
        return gesture


    return "negative"
    
  def detect_gesture(self, pose):
    # pose: the newly detected pose
    # start_end_matches: dict of start pose and end pose pairs to match dynamic gestures
    if len(self.buffer) == 0:
      self.buffer.append(pose)
      return "negative"
    
    # avoid detecting duplicated pose
    if self.buffer[-1] == pose:
      return "negative"
    
    # find match for key pose sequences
    self.buffer.append(pose)
    matched_gesture = self.find_start_end_matches(self.buffer[-3:])
    # if find a match, empty buffer and return the gesture name
    if matched_gesture != -1:
      self.buffer = []
      return matched_gesture
    
    # if there is no match, return negative class
    return "negative"

  def to_degree(self, rad):
    return (rad * 180) / math.pi

  def detect_pose(self, feature, thres=0.5):
    yaw, pitch, roll, handedness = feature[-4:]
    
    # feature vector X
    # X = np.array(feature).reshape(1,-1)
    pose, score = self.classifier.predict_proba(feature)
    # pose_idx = np.argmax(pred)
    # score = np.max(pred)

    if pose == "palm":
      # heuristic check for the directional poses
      yaw = self.to_degree(yaw)
      pitch = self.to_degree(pitch)
      roll = self.to_degree(roll)

      if (yaw <= -20.0 and handedness == 0) or (yaw <= -15.0 and handedness == 1):
        return "left"
      elif (yaw >= 15.0 and handedness == 0) or (yaw >= 20.0 and handedness == 1):
        return "right"
      elif pitch >= 20.0:
        return "up"
      elif pitch <= -25.0:
        return "down"
      elif roll >= 110.0 and handedness == 0 :
        return "palm_r"
      elif roll <= -110.0 and handedness == 1:
        return "palm_l"
      else:
        return "palm"

    else:
      return "fist"
  
  def find_start_end_matches(self, lst_poses):
    # print(lst_poses)
    matched_gesture = -1
    for gesture in START_END_MATCHES_LABEL:
      key_sequences = START_END_MATCHES_LABEL[gesture]
      # loop through all possible key sequences of a gesture
      for sequence in key_sequences:
        # the sequence length larger than list of detected pose
        if len(sequence) > len(lst_poses):
          continue
        # find the occurence of the key sequence in the detected pose
        elif lst_poses[-len(sequence):] == sequence:
          matched_gesture = gesture
          break
      
    return matched_gesture