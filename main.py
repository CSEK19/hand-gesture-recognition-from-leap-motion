################################################################################
# Copyright (C) 2012-2018 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
################################################################################

import Leap, sys, _thread, time
import LeapPython
import Leap
from model_heuristic import StaticHandPoseClassifier, HandGestureRecognizer
from utils import Visualizer
import cv2
import pyautogui
# from pywinauto.keyboard import send_keys
import keyboard
import glob, os, json

def extract_feature(frame):
  hand = frame.hands[0]
  feature = []
  # extract skeleton: palm, wrist, thumb(3), index(4), middle(4), ring(4), pinky(4)
  feature += [hand.palm_position.x, hand.palm_position.y, hand.palm_position.z]
  
  # handedness, yaw, pitch, roll
  handedness = 0 if hand.is_left else 1
  yaw, pitch, roll = hand.direction.yaw, hand.direction.pitch, hand.palm_normal.roll

  # add fingers' skeleton
  for finger in hand.fingers:
    # Get bones
    for b in range(0, 4):
      bone = finger.bone(b)
      if b == 0:
        feature += [bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z]

      feature += [bone.next_joint.x, bone.next_joint.y, bone.next_joint.z]

  feature += [yaw, pitch, roll, handedness]
  return feature


gesture_key_map = {
  "move left": "left",
  "move right": "right",
  "move up": "up",
  "move down": "down",
  "rotate left": "z",
  "rotate right": "z",
  "thumb in": "p",
  "stop": "b",
  "close fist": "enter"
}


def game_control(gesture):
  # if gesture == "close fist":
  #   keyboard.press_and_release('s')
  if gesture in gesture_key_map:
    key = gesture_key_map[gesture]
    # print(key)
    if key in ["left", "right", "up", "down"]:
      pyautogui.press(key)
    else:
      keyboard.press_and_release(key)

def run_HGR():
  # print("Start process 1")
  # init leap controller
  controller = Leap.Controller()
  controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
  record = True
  # init hand gesture recognizer and visualizer
  static_model_weight = 'model\\weights\\SVC_weights_palmfist_2204.pkl'
  scaler_weight = 'model\\weights\\stdscaler_weights_palmfist_2204.pkl'
  vis = Visualizer()
  static_classifier = StaticHandPoseClassifier(static_model_weight, scaler_weight)
  recognizer = HandGestureRecognizer(static_classifier)
  
  recorded_frames = []
  recorded_data = []
  frame_cnt = 0
  out = 'sth'
  if not os.path.exists(out):
      os.makedirs(out)

  while True:
    frame = controller.frame()
    image = frame.images[0]

    if image.is_valid:
      hand_feature = []
      display = None
      if not frame.hands.is_empty:
        # make detection
        hand_feature = extract_feature(frame)
        gesture = recognizer.detect(hand_feature)
        # print(gesture)
        if gesture != 'negative':
          display = gesture
          game_control(gesture)

        # record
        recorded_frames.append(vis.visualize(frame.images))
        recorded_data.append(hand_feature)
        frame_cnt += 1


        # save every fra
        # me_cnt
        if frame_cnt == 50000 or keyboard.is_pressed("a"):
          print("Start recording ...")
          assert len(recorded_frames) == len(recorded_data)
          frame_cnt = 0
          last_idx = len(glob.glob(f'{out}/*.mp4'))
          # save json file
          with open(f'{out}/{last_idx}.json', 'w') as f:
            json.dump(recorded_data, f)
          # save clip
          vid = cv2.VideoWriter(f'{out}/{last_idx}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (400, 400), 0)
          for frame in recorded_frames:
            vid.write(frame)
          vid.release()
          # empty the recorded
          recorded_frames = []
          recorded_data = []
          print("End recording ...")



      # visualize
      # vis_img = vis.visualize(frame.images, display)

      # cv2.imshow('LeapDemo', vis_img)
      # if cv2.waitKey(1) == ord('q'):
      #   break


if __name__ == "__main__":
  run_HGR()
