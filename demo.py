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

static_model_weight = 'model\\weights\\SVC_weights_palmfist_2204.pkl'
scaler_weight = 'model\\weights\\stdscaler_weights_palmfist_2204.pkl'

def game_control(gesture):
  if gesture == "move left":
    pyautogui.keyDown('left')
    pyautogui.keyUp('left')
  elif gesture == "move right":
    pyautogui.keyDown('right')
    pyautogui.keyUp("right")
  elif gesture == "close fist":
    pyautogui.keyDown("enter")
    pyautogui.keyUp("enter")    
  elif gesture == "rotate":
    pyautogui.keyDown("z")
    pyautogui.keyUp("z")

def run():
  # init leap controller
  controller = Leap.Controller()
  controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
  
  # init hand gesture recognizer and visualizer
  vis = Visualizer()
  static_classifier = StaticHandPoseClassifier(static_model_weight, scaler_weight)
  recognizer = HandGestureRecognizer(static_classifier)

  while True:

    frame = controller.frame()
    image = frame.images[0]

    if image.is_valid:
      display = None
      if not frame.hands.is_empty:
        # make detection
        hand_feature = extract_feature(frame)
        gesture = recognizer.detect(hand_feature)
        # print(gesture)
        if gesture != 'negative':
          display = gesture
          game_control(gesture)

      # visualize
      vis_img = vis.visualize(frame.images, display)
      cv2.imshow('LeapDemo', vis_img)
      if cv2.waitKey(1) == ord('q'):
        break



def main():
  # Keep this process running until Enter is pressed
  print("Press Enter to quit...")
  try:
    run()
  except KeyboardInterrupt:
    sys.exit(0)

if __name__ == "__main__":
    main()



