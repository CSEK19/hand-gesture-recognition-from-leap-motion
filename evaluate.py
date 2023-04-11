import pickle
import numpy as np
from model import StaticHandPoseClassifier
from tqdm import tqdm
import logging
import cv2
import os
import math

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('evaluate.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

poses = ['palm', 'fist', 'stop', 'thumb_in', 'left', 'right', 'up', 'down', 'rotate']
gestures = ['close fist', 'move left', 'move right', 'move up', 'move down', 'rotate', 'stop', 'thumb_in', 'negative']
start_end_label_matches = {'move down': ['palm', 'down'], \
                            'move up': ['palm','up'], \
                            'move left': ['palm','left'], \
                            'move right': ['palm', 'right'], \
                            'close fist': ['palm', 'fist'], \
                            'rotate': ['palm', 'rotate'], \
                            'stop': ['palm', 'stop'], \
                            'thumb_in': ['palm', 'thumb_in']}

start_end_matches = {}
for gesture in start_end_label_matches:
    gesture_id = gestures.index(gesture)
    start_end_matches[gesture_id] = []
    [start_end_matches[gesture_id].append(poses.index(x)) for x in start_end_label_matches[gesture]]


def detect_gesture(prev_pose_id, pose_id, start_end_matches):
    # prev_pose_id: the previous pose
    # pose_id: the newly detected pose
    # start_end_matches: dict of start pose and end pose pairs to match dynamic gestures
    for gesture_id in start_end_matches:
        pair = start_end_matches[gesture_id]
        if pair[0] == prev_pose_id and pair[1] == pose_id:
            return gesture_id
    return -1


def detect_pose(feature, detector, thres=0.5):
  yaw, pitch, roll, handedness = feature[-4:]
  # feature vector X
  X = np.array(feature).reshape(1,-1)
  pred = detector.predict_proba(X)[0]
  pose_idx = np.argmax(pred)
  score = np.max(pred)

  # X = np.array(feature).reshape(1,-1)
  # pose_idx = detector.predict(X)[0]
  # score = 0.5

  if score < thres:
    return -1

  if poses[pose_idx] == "palm":
    if yaw >= math.pi/6:
      return poses.index("right")
    elif yaw <= -math.pi/6:
      return poses.index("left")
    else:
      if pitch >= math.pi/6:
        return poses.index("up")
      elif pitch <= -math.pi/6:
        return poses.index("down")
      else:
        if (handedness == 1 and roll <= -math.pi/2) or (handedness == 0 and roll >= math.pi/2):
          return poses.index("rotate")
        else:
          return pose_idx
  
  elif poses[pose_idx] == "fist":
    return pose_idx
  elif poses[pose_idx] == "stop":
    if (handedness == 1 and (-math.pi/6)) >= roll or (handedness == 0 and (math.pi/6) <= roll):
      return -1
    return pose_idx
  elif poses[pose_idx] == "thumb_in":
    return pose_idx
  
  return -1

def debug(path, preds):
  src_path = path
  vid = cv2.VideoCapture(src_path)
  out = 'Debug/' + path
  if not os.path.exists(out):
    os.makedirs(out)

  out_vid = cv2.VideoWriter(f'{out}/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (400, 400), 0)
  # print(len(preds), src_path, vid.get(cv2.CAP_PROP_FRAME_COUNT))
  assert len(preds) == vid.get(cv2.CAP_PROP_FRAME_COUNT)
  for pose_id in preds:
    _, vis_img = vid.read()
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
    if pose_id != '?':
      vis_img = cv2.putText(vis_img, f'{poses[pose_id]}', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
    out_vid.write(vis_img)
  out_vid.release()

def predict(X, detector):
  predictions = []

  prev_pose_id, pose_id = -1, -1
  # X: list of skeletons of a video clip
  
  for skeleton in X:
    if len(skeleton) == 0:
      continue

    # Make detection
    pose_id = detect_pose(skeleton, detector)
    # print(pose_id)
    if pose_id != -1:
      # print(prev_pose_id, pose_id)
      gesture_id = detect_gesture(prev_pose_id, pose_id, start_end_matches)
      if gesture_id != -1:
        predictions.append(gesture_id)

      prev_pose_id = pose_id

  return predictions
        


def main():
  detections = {}
  # init detector
  detector = StaticHandPoseClassifier('weights\\MLPClassifier_0411.pkl', 'weights\\StandardScaler_0411.pkl')

  # load evaluatation dataset
  with open('array_dynamic_0404.pkl', 'rb') as f:
    data = pickle.load(f)

  len_data = len(data)
  correct = 0
  for X, y, path in tqdm(data, position=0, leave=True):
    if y not in detections:
      detections[y] = {'correct': 0, 'cnt': 0}
    detections[y]['cnt'] += 1
    pred = predict(X, detector)
    # print(pred)
    # Predict only 1 correct gesture -> accurate

    if (len(pred) == 0 and y == gestures.index('negative')) or (len(pred) == 1 and pred[0] == y):
      detections[y]['correct'] += 1
      correct += 1
    # else: # debug
    #   logger.info(f'{gestures[y]}, {path}: {[gestures[x] for x in pred]}')
    #   debug(path, pose_pred)

  print(f'Accuracy: {correct/len_data}')
  for y in detections:
    name = gestures[y]
    correct, cnt = detections[y]["correct"], detections[y]["cnt"]
    print(f'{name}: {correct}/{cnt}')


if __name__ == "__main__":
  main()