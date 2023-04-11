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

poses = ['palm', 'fist', 'stop', 'left', 'right', 'up', 'down', 'rotate', 'thumb_in']
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
    


def run(controller, detector):
    # for display purpose
    n_frames_displayed = 30
    display_cnt = 99999
    # initialization
    maps_initialized = False
    # negative_id = poses.index("negative")
    prev_pose_id, pose_id = -1, -1 # set the initial pose_ids to negative class
    # to check pose count reach a minimum frame threshold or not
    prev_tmp_pose = -1
    min_frame_thres = 10
    tmp_cnt = 0

    gesture_id = -1

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
            hand = frame.hands[0]
            skeleton = []
            # extract skeleton: palm, wrist, thumb(3), index(4), middle(4), ring(4), pinky(4)
            skeleton += [hand.palm_position.x, hand.palm_position.y, hand.palm_position.z]
            skeleton += [hand.arm.wrist_position.x, hand.arm.wrist_position.y, hand.arm.wrist_position.z]
            handedness = 0 if hand.is_left else 1

            # add fingers' skeleton
            for finger in hand.fingers:
                # Get bones
                for b in range(0, 4):
                    bone = finger.bone(b)
                    # hand_data['skeleton'] += [bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z]
                    joint = (bone.prev_joint + bone.next_joint) / 2
                    skeleton += [joint.x, joint.y, joint.z]

            skeleton += [hand.direction.yaw, hand.direction.pitch, hand.palm_normal.roll, handedness]

            # Make detection
            # print(skeleton)
            X = np.array(skeleton).reshape(1,-1)
            pred = detector.predict_proba(X)[0] # score array
            tmp_pose = np.argmax(pred)
            score = np.max(pred)
            
            # tmp_pose = detector.predict(X)[0]
            # score = ""
            vis_img = cv2.putText(vis_img, f'{poses[tmp_pose]}:{score}', (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)

            '''
            if tmp_pose != negative_id and tmp_pose != pose_id:
                tmp_cnt = 0 if (tmp_pose != prev_tmp_pose) else (tmp_cnt+1)
                prev_tmp_pose = tmp_pose
                # accept the pose if the count reach the minimum threshold
                if tmp_cnt >= min_frame_thres:
                    pose_id = tmp_pose
                    # detect the dynamic gesture
                    gesture_id = detectGesture(prev_pose_id, pose_id, start_end_matches)
                    # print(prev_pose_id,pose_id,gesture)
                    prev_pose_id = pose_id
                    # display the dynamic gesture in several frames
                    if gesture_id != -1:
                        display_cnt = 0
                        print(gestures[gesture_id])
                        # vis_img = cv2.putText(vis_img, gestures[gesture], (200,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            '''
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
    detector = StaticHandPoseClassifier('weights\\LogisticRegression_0404.pkl', 'weights\\StandardScaler_0404.pkl')

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        run(controller, detector)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()



