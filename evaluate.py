import pickle
import numpy as np
from model import StaticHandPoseClassifier, HandGestureRecognizer
from tqdm import tqdm
import logging
import cv2
import os
import math
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

GESTURES = ['close fist', 'move left', 'move right', 'move up', 'move down', 'rotate left', 'rotate right', 'stop', 'thumb in', 'negative']

def debug(path, X, recognizer, static_classifier):
  vid = cv2.VideoCapture(path + '/video.mp4')
  out = 'Debug/' + path
  if not os.path.exists(out):
    os.makedirs(out)

  recognizer.clear_buffer()

  debug_vid = cv2.VideoWriter(f'{out}/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (400, 400), 0)
  for hand_feature in X:
    _, vis_img = vid.read()
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
    # detect
    if hand_feature != []:
      yaw, pitch, roll, _ = hand_feature[-4:]
      yaw = int((yaw * 180) / math.pi)
      pitch = int((pitch * 180) / math.pi)
      roll = int((roll * 180) / math.pi)

      gesture = recognizer.detect(hand_feature)
      pose, score = recognizer.detect_pose(hand_feature, heuristic=True, thres=0.45)
      pose_class, score_class = static_classifier.predict_proba(hand_feature)

      vis_img = cv2.putText(vis_img, f'{gesture}', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
      vis_img = cv2.putText(vis_img, f'{pose}:{score}', (100,125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
      vis_img = cv2.putText(vis_img, f'{pose_class}:{score_class}', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
      vis_img = cv2.putText(vis_img, f'{yaw},{pitch},{roll}', (100,175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    debug_vid.write(vis_img)

  debug_vid.release()

def predict(X, recognizer):
  elapse_times = 0
  n_frames = 0

  recognizer.clear_buffer()
  pred = []
  for hand_feature in X:
    # no hand detected, skip
    if len(hand_feature) == 0:
      continue

    n_frames += 1
    start = time.perf_counter()
    gesture = recognizer.detect(hand_feature, heuristic=False)
    end = time.perf_counter()
    elapse_times += (end-start)

    if gesture != "negative":
      pred.append(gesture)

  # average alapse frame for each frame
  elapse = elapse_times / n_frames
  return pred, elapse
        

def main():
  detections = {}
  correct = 0

  # for confusion matrix
  y_true = []
  y_pred = []
  negative_idx = GESTURES.index('negative')
  n = 0
  for gesture in GESTURES:
    detections[gesture] = {'TP': 0, 'FP': 0, 'FN': 0, 'n': 0}
  # init hand gesture recognizer and visualizer
  static_model_weight = 'model\\weights\\SVC_weights_0805.pkl'
  scaler_weight = 'model\\weights\\scaler_weights_0805.pkl'
  static_classifier = StaticHandPoseClassifier(static_model_weight, scaler_weight)
  recognizer = HandGestureRecognizer(static_classifier)

  # load evaluation dataset
  with open('data_collection/array_data/gesture_data_v5.pkl', 'rb') as f:
    data = pickle.load(f)

  elapse_times = []
  n_samples = len(data)

  for X, y, path in tqdm(data, position=0, leave=True):
    gesture = GESTURES[y]
    if gesture != 'negative':
      n += 1
    detections[gesture]['n'] += 1

    # make prediction on the video
    pred, elapse = predict(X, recognizer)

    elapse_times.append(elapse)

    wrong = False
    # If any prediction is made in negative samples, add FP for each class
    if gesture == 'negative':
      if len(pred) == 0:
        # for cf matrix
        y_true.append(negative_idx)
        y_pred.append(negative_idx)
      else:
        for p in pred:
          wrong = True
          detections[p]['FP'] += 1
          # for cf matrix
          y_true.append(negative_idx)
          y_pred.append(GESTURES.index(p))
    # If the ground truth is not negative class, update TP, FP, FN

    else:
      ges_found = False
      for p in pred:
        y_pred.append(GESTURES.index(p))
        if p == gesture and not ges_found:
          ges_found = True
          correct += 1
          detections[p]['TP'] += 1
          # for cf matrix
          y_true.append(y)
        else:
          detections[p]['FP'] += 1
          wrong = True
          # for cf matrix
          y_true.append(negative_idx)
      # if there is no gesture found, add FN
      if not ges_found:
        detections[gesture]['FN'] += 1
        wrong = True
        # for cf matrix
        y_true.append(y)
        y_pred.append(negative_idx)


    if wrong: # debug if there are any wrong cases
      debug(path, X, recognizer, static_classifier)

  overall_acc = accuracy_score(y_true, y_pred)
  print(f'Overall accuracy: {overall_acc}')

  # Precision and recall for each class
  precisions = []
  recalls = []
  f1_scores = []
  weighted_sum_precisions = 0.0
  weighted_sum_recalls = 0.0
  weighted_sum_f1_scores = 0.0

  for gesture in GESTURES[:-1]:
    # calculate precision, recall and f1 score
    TP, FP, FN = detections[gesture]["TP"], detections[gesture]["FP"], detections[gesture]["FN"]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = (2*precision*recall)/(precision+recall)

    # update weighted sum
    weight = detections[gesture]['n']/n
    weighted_sum_precisions += weight * precision
    weighted_sum_recalls += weight * recall
    weighted_sum_f1_scores += weight * f1_score

    # update precision and recall list
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)


  # append weighted average result
  precisions.append(weighted_sum_precisions)
  recalls.append(weighted_sum_recalls)
  f1_scores.append(weighted_sum_f1_scores)
  name = GESTURES[:-1] + ['Weighted average']

  # display result table
  df = pd.DataFrame({'Gesture': name, 'Precision': precisions, 'Recall': recalls, 'F1 score': f1_scores})
  # df.to_csv('gesture_evaluation.csv')
  print(df)

  print('Average elapse time for each frame in the whole dataset:', sum(elapse_times)/n_samples)

  # cf matrix:
  cf_matrix = confusion_matrix(y_true, y_pred)
  cf_matrix_norm = np.around(cf_matrix / cf_matrix.sum(axis=1)[:,None], 2)
  sns.set(font_scale=1.5)
  sns.set(rc={"figure.figsize":(15, 15)})
  ax = sns.heatmap(cf_matrix_norm, annot=True, cmap='Blues')
  ax.set_title(f'Confusion matrix, accuracy: {round(overall_acc * 100, 2)}%', fontsize=20)
  ax.set_xlabel('\nPredicted Values', fontsize=20)
  ax.set_ylabel('Actual Values', fontsize=20)

  gesture_display = ['close\nfist  ','move\nleft  ','move\nright ','move\nup    ','move\ndown','rotate\nleft   ','rotate\nright  ','stop','thumb\nin      ', 'negative']
  ax.xaxis.set_ticklabels(gesture_display, fontsize=15)
  ax.yaxis.set_ticklabels(gesture_display, fontsize=15)
  plt.show()
  # plt.savefig("D:\\bachkhoa\\bachkhoa222\\Luan van\\images\\training_result\\gesture_cf_matrix.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
  main()