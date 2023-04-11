################################################################################
# Copyright (C) 2012-2018 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
################################################################################

import Leap, sys, _thread, time
import LeapPython
import cv2, Leap, math, ctypes
import numpy as np
import os, glob, json
from model import StaticHandPoseClassifier
from visualize import convert_distortion_maps, undistort
import pickle

poses = ['palm', 'fist', 'stop', 'thumb_in', 'left', 'right', 'up', 'down', 'rotate']
gestures = ['move down', 'close fist', 'move left', 'move right', 'rotate', 'move up', 'negative']
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

def extract_feature(frame):
  image = frame.images[0]
  feature = []
  # extract skeleton: palm, wrist, thumb(3), index(4), middle(4), ring(4), pinky(4)
  feature += [hand.palm_position.x, hand.palm_position.y, hand.palm_position.z]
  feature += [hand.arm.wrist_position.x, hand.arm.wrist_position.y, hand.arm.wrist_position.z]
  # handedness, yaw, pitch, roll
  handedness = 0 if hand.is_left else 1
  yaw, pitch, roll = hand.direction.yaw, hand.direction.pitch, hand.palm_normal.roll

  # add fingers' skeleton
  for finger in hand.fingers:
    # Get bones
    for b in range(0, 4):
      bone = finger.bone(b)
      # hand_data['skeleton'] += [bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z]
      joint = (bone.prev_joint + bone.next_joint) / 2
      feature += [joint.x, joint.y, joint.z]

  feature += [yaw, pitch, roll, handedness]
  return feature

def detect_pose(feature, detector, thres=0.5):
  yaw, pitch, roll, handedness = feature[-4:]
  # feature vector X
  X = np.array(skeleton).reshape(1,-1)
  pred = detector.predict_proba(X)[0]
  pose_idx = np.argmax(pred)
  score = np.max(pred)

  if score < thres:
    return -1

  if poses[pose_idx] == "palm":
    if yaw >= math.pi/4:
      return poses.index("right")
    elif yaw <= -math.pi/4:
      return poses.index("left")
    else:
      if pitch >= math.pi/4:
        return poses.index("up")
      elif pitch <= -math.pi/4:
        return poses.index("down")
      else:
        if (handedness == 1 and roll <= -math.pi/4) or (handedness == 0 and roll >= math.pi/4):
          return poses.index("rotate")
        else:
          return pose_idx
  
  elif poses[pose_idx] == "fist":
    return pose_idx
  elif poses[pose_idx] == "stop":
    if (handedness == 1 and roll <= (-math.pi/4)) or (handedness == 0 and roll >= (math.pi/4)):
      return "-1
    return pose_idx
  elif poses[pose_idx] == "thumb_in":
    return pose_idx
  
  return -1


def run(controller, detector):
  # initialization
  maps_initialized = False
  # negative_id = poses.index("negative")
  prev_pose_id, pose_id = -1, -1 # set the initial pose_ids to negative class


  gesture_id = -1

  while True:
    frame = controller.frame()

    # record if image is valid
    if not frame.is_valid:
      continue

    if not maps_initialized:
      left_coordinates, left_coefficients = convert_distortion_maps(frame.images[0])
      right_coordinates, right_coefficients = convert_distortion_maps(frame.images[1])
      maps_initialized = True

    vis_img = undistort(image, left_coordinates, left_coefficients, 400, 400)
    # undistorted_right = undistort(image, right_coordinates, right_coefficients, 400, 400)

    if not frame.hands.is_empty:
      feature = extract_feature(frame)

      # Make detection
      pose_id = predict(feature, detector)
    
      if pose_id != -1:
        prev_pose_id = pose_id

      gesture_id = detect_gesture(prev_pose_id, pose_id, start_end_matches)
      print(gestures[gesture_id])
      
      # visualize
      vis_img = cv2.putText(vis_img, f'{poses[pose_id]}:{score}', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)


      # if gesture_id != -1 and display_cnt < n_frames_displayed:
      #     display_cnt += 1
      #     vis_img = cv2.putText(vis_img, gestures[gesture_id], (200,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
        
    cv2.imshow('LeapDemo', vis_img)
    if cv2.waitKey(1) == ord('q'):
        break


def main():
    # Create a sample listener and controller
    # listener = SampleListener()
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
    detector = StaticHandPoseClassifier('weights\\LogisticRegression_0411.pkl', 'weights\\StandardScaler_0404.pkl')

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        run(controller, detector)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()



