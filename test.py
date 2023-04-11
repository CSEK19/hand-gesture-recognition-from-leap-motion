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
import math

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



def run(controller):
    # for display purpose
    n_frames_displayed = 30
    display_cnt = 99999
    # initialization
    maps_initialized = False
    negative_id = poses.index("negative")
    prev_pose_id, pose_id = negative_id,negative_id # set the initial pose_ids to negative class
    # to check pose count reach a minimum frame threshold or not
    prev_tmp_pose = -1
    min_frame_thres = 10
    tmp_cnt = 0

    gesture_id = -1
    
    cnt = 0
    while True:
        frame = controller.frame()
        image = frame.images[0]

        # record if image is valid
        if not image.is_valid:
            continue

        if not maps_initialized:
            left_coordinates, left_coefficients = convert_distortion_maps(frame.images[0])
            right_coordinates, right_coefficients = convert_distortion_maps(frame.images[1])
            maps_initialized = True


        vis_img = undistort(image, left_coordinates, left_coefficients, 400, 400)
        # undistorted_right = undistort(image, right_coordinates, right_coefficients, 400, 400)

        if not frame.hands.is_empty:
            cnt += 1
            hand = frame.hands[0]
            skeleton = []
            # extract skeleton: palm, wrist, thumb(3), index(4), middle(4), ring(4), pinky(4)
            skeleton += [hand.palm_position.x, hand.palm_position.y, hand.palm_position.z]
            skeleton += [hand.arm.wrist_position.x, hand.arm.wrist_position.y, hand.arm.wrist_position.z]
            handedness = 0 if hand.is_left else 1

            # add fingers' skeleton
            for idx,finger in enumerate(hand.fingers):
                # Get bones
                for b in range(0, 4):
                    bone = finger.bone(b)
                    # hand_data['skeleton'] += [bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z]
                    joint = (bone.prev_joint + bone.next_joint) / 2
                    skeleton += [joint.x, joint.y, joint.z]
                    if idx == 1 and b == 3:
                        index_tip = [joint.x, joint.y, joint.z]
                    elif idx == 2 and b == 3:
                        middle_tip = [joint.x, joint.y, joint.z]

            skeleton += [hand.direction.yaw, hand.direction.pitch, hand.direction.roll, handedness]
            if cnt & 50 == 0:
                # print(f"Yaw, pitch, roll: {hand.direction.yaw}, {hand.direction.pitch}, {hand.palm_normal.roll}")
                dist = math.sqrt((index_tip[0]-middle_tip[0])**2 + (index_tip[1]-middle_tip[1])**2 + (index_tip[2]-middle_tip[2])**2)
                print("index and middle tip diff: ", dist)

            
        cv2.imshow('LeapDemo', vis_img)
        if cv2.waitKey(1) == ord('q'):
            break


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


