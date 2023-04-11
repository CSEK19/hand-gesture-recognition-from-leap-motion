import pickle
import numpy as np
from model import StaticHandPoseClassifier
from tqdm import tqdm
import logging
import cv2
import os

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('evaluate.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

poses = ['down', 'fist', 'left', 'up', 'palm', 'right', 'rotate','negative']
gestures = ['move down', 'close fist', 'move left', 'move right', 'rotate', 'move up', 'negative']
start_end_label_matches = {'move down': ['palm', 'down'], \
                            'move up': ['palm','up'], \
                            'move left': ['palm','left'], \
                            'move right': ['palm', 'right'], \
                            'close fist': ['palm', 'fist'], \
                            'rotate': ['palm', 'rotate']}

start_end_matches = {}
for gesture in start_end_label_matches:
    gesture_id = gestures.index(gesture)
    start_end_matches[gesture_id] = []
    [start_end_matches[gesture_id].append(poses.index(x)) for x in start_end_label_matches[gesture]]

def detectGesture(prev_pose_id, pose_id, start_end_matches):
    # prev_pose_id: the previous pose
    # pose_id: the newly detected pose
    # start_end_matches: dict of start pose and end pose pairs to match dynamic gestures
    for gesture_id in start_end_matches:
        pair = start_end_matches[gesture_id]
        if pair[0] == prev_pose_id and pair[1] == pose_id:
            return gesture_id
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
  pose_predictions = []
  prev_tmp_pose = -1
  min_frame_thres = 10
  tmp_cnt = 0

  negative_id = poses.index('negative')
  prev_pose_id, pose_id = negative_id, negative_id
  # X: list of skeletons of a video clip
  
  for skeleton in X:
    if len(skeleton) != 70:
      pose_predictions.append('?')
      continue
    inp = np.array(skeleton).reshape(1,-1)
    pred = detector.predict_proba(inp)[0] # score array
    # tmp_pose = detector.predict(X)[0]
    tmp_pose = np.argmax(pred)
    pose_predictions.append(tmp_pose)
    score = np.max(pred)

    if tmp_pose != negative_id and tmp_pose != pose_id:
        tmp_cnt = 0 if (tmp_pose != prev_tmp_pose) else (tmp_cnt+1)
        prev_tmp_pose = tmp_pose
        # accept the pose if the count reach the minimum threshold
        if tmp_cnt >= min_frame_thres:
            pose_id = tmp_pose
            # detect the dynamic gesture
            gesture_id = detectGesture(prev_pose_id, pose_id, start_end_matches)
            prev_pose_id = pose_id
            if gesture_id != -1:
              predictions.append(gesture_id)

  return predictions, pose_predictions
        


def main():
  detections = {}
  # init detector
  detector = StaticHandPoseClassifier('weights\SVC_v2.pkl')

  # load evaluatation dataset
  with open('array_dynamic.pkl', 'rb') as f:
    data = pickle.load(f)

  len_data = len(data)
  correct = 0
  for X, y, path in tqdm(data, position=0, leave=True):
    if y not in detections:
      detections[y] = {'correct': 0, 'cnt': 0}
    detections[y]['cnt'] += 1
    pred, pose_pred = predict(X, detector)
    # print(pred)
    # Predict only 1 correct gesture -> accurate

    if (len(pred) == 0 and y == gestures.index('negative')) or (len(pred) == 1 and pred[0] == y):
      detections[y]['correct'] += 1
      correct += 1
    else: # debug
      logger.info(f'{gestures[y]}, {path}: {[gestures[x] for x in pred]}')
      debug(path, pose_pred)

  print(f'Accuracy: {correct/len_data}')
  for y in detections:
    name = gestures[y]
    correct, cnt = detections[y]["correct"], detections[y]["cnt"]
    print(f'{name}: {correct}/{cnt}')
  # for y in data:
    # print(f'Accuracy {gestures[y]}: {detections[y]["correct"]}/{detections[y]["cnt"]}')

if __name__ == "__main__":
  main()