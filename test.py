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
from utils import Visualizer
import pickle
import math


def angle(v1, v2):
  unit_v1 = v1 / np.linalg.norm(v1)
  unit_v2 = v2 / np.linalg.norm(v2)
  dot_product = abs(np.dot(unit_v1, unit_v2))
  angle = np.arccos(dot_product)
  return angle

def distance(x1, x2):
    return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2 + (x1[2] - x2[2])**2)

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
def run(controller):
    vis = Visualizer()
    cnt = 0
    idx= 0 
    while True:
        frame = controller.frame()
        image = frame.images[0]

        # record if image is valid
        if not image.is_valid:
            continue

        for hand in frame.hands:
            hand_data = {}
            hand_data['handedness'] = "Left" if hand.is_left else "Right"
            hand_data['direction'] = [hand.direction.x, hand.direction.y, hand.direction.z]
            hand_data['normal'] = [hand.palm_normal.x, hand.palm_normal.y, hand.palm_normal.z]
            hand_data['roll'] = hand.palm_normal.roll
            hand_data['pitch'] = hand.direction.pitch
            hand_data['yaw'] = hand.direction.yaw
            hand_data['wrist'] = [hand.arm.wrist_position.x, hand.arm.wrist_position.y, hand.arm.wrist_position.z]

            # extract skeleton: palm, thumb(4), index(5), middle(5), ring(5), pinky(5)
            hand_data['palm'] = [hand.palm_position.x, hand.palm_position.y, hand.palm_position.z]
            palm = hand_data['palm']
            for finger in hand.fingers:
                # print(finger.id, finger.type)
                hand_data[FINGERS[finger.type]] = []
                # Get bones
                for b in range(0, 4):
                    bone = finger.bone(b)
                    if b == 0:
                        hand_data[FINGERS[finger.type]] += [bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z]
                    hand_data[FINGERS[finger.type]] += [bone.next_joint.x, bone.next_joint.y, bone.next_joint.z]
            
            # thumb_angles
            thumb = np.array(hand_data["thumb"]).reshape(-1,3)
            thumb_tip = thumb[-1]
            # print(thumb.shape)
            thumb_bones = [thumb[i+1]-thumb[i] for i in range(1,4)]
            thumb_angles = np.array([angle(thumb_bones[i], thumb_bones[i+1]) for i in range(0,2)])
            print(thumb_angles * 180 / math.pi)
            # print(distance(thumb_tip, palm))
            # print(thumb_tip[1] - palm[1])


            index = np.array(hand_data["index"]).reshape(-1,3)
            index_tip = index[-1]
            middle = np.array(hand_data["middle"]).reshape(-1,3)
            middle_tip = middle[-1]
            
            pinky = np.array(hand_data["pinky"]).reshape(-1,3)
            pinky_bones = [pinky[i+1]-pinky[i] for i in range(0,4)]
            # print(angle(pinky_bones[1], pinky_bones[2]) * 180 / math.pi)
            # print(hand_data['pitch'] * 180 / math.pi)
            # print(thumb[-1][0], index_tip[0], middle_tip[0])


            # print(distance(thumb_tip, pinky[1]))
            # print(hand_data['pitch'] * 180 / math.pi)

        vis_img = vis.visualize(frame.images)
        cv2.imshow('LeapDemo', vis_img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('s'):
            print('Writing img')
            cv2.imwrite(f'vis_img/{idx}.jpg', vis_img)
            idx += 1



def main():
    # Create a sample listener and controller
    # listener = SampleListener()
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        run(controller)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()


