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

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
def convert_distortion_maps(image):

    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length//2, dtype=np.float32)
    ymap = np.zeros(distortion_length//2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length//2 - i//2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length//2 - i//2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width//2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width//2))

    #resize the distortion map to equal desired destination image size
    resized_xmap = cv2.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)
    resized_ymap = cv2.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)

    #Use faster fixed point maps
    coordinate_map, interpolation_coefficients = cv2.convertMaps(resized_xmap,
                                                                 resized_ymap,
                                                                 cv2.CV_32FC1,
                                                                 nninterpolation = False)

    return coordinate_map, interpolation_coefficients

def undistort(image, coordinate_map, coefficient_map, width, height):
    destination = np.empty((width, height), dtype = np.ubyte)
    # ptr = Leap.byte_array(image.width * image.height * image.bytes_per_pixel)
    ptr = image.data_method()
    # print(ptr)
    # print(hex(id(ptr)))
    #wrap image data in numpy array
    
    i_address = (int(ptr.cast()))

    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    # as ctypes array
    as_ctype_array = ctype_array_def.from_address(i_address)
    # as numpy array
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    #remap image to destination
    destination = cv2.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation = cv2.INTER_LINEAR)

    #resize output to desired destination size
    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination

DYNAMIC_GESTURES = ['d_fist', 'd_rotate_left', 'd_rotate_right', 'd_left', 'd_right', 'd_up', 'd_down', 'd_negative', 'd_stop', 'd_thumb']
STATIC_GESTURES = ['palm', 'palm_left', 'palm_right', 'palm_up', 'up', 'down', 'left', 'right', 'fist', 'hook', 'stop', 'thumb_in', 'negative']

def run(controller):
    maps_initialized = False

    out_dir = 'data_collection/LeapDatav4/Xuan'
    dynamic = True
    recording = False
    stop_flag = False

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


        undistorted_left = undistort(image, left_coordinates, left_coefficients, 400, 400)
        undistorted_right = undistort(image, right_coordinates, right_coefficients, 400, 400)

        cv2.imshow('Left', undistorted_left)

        # recording skeleton and depth frame
        if recording:
            if frame_id == 150 and dynamic==False:
                stop_flag = True
            else:
                if frame_id > 0 and recorded_data[frame_id-1]['origin_frame'] == frame.id:
                    pass
                else:
                    recorded_frames.append(undistorted_left)
                    recorded_data[frame_id] = {}

                    recorded_data[frame_id]['origin_frame'] = frame.id
                    recorded_data[frame_id]['hands'] = []

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
                        for finger in hand.fingers:
                            # print(finger.id, finger.type)
                            hand_data[FINGERS[finger.type]] = []
                            # Get bones
                            for b in range(0, 4):
                                bone = finger.bone(b)
                                if b == 0:
                                    hand_data[FINGERS[finger.type]] += [bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z]
                                hand_data[FINGERS[finger.type]] += [bone.next_joint.x, bone.next_joint.y, bone.next_joint.z]


                        # print(len(hand_data['skeleton']))
                        recorded_data[frame_id]['hands'].append(hand_data)
                    frame_id += 1
  

        # keyboard input for recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            print('Start recording')
            recording = True
            frame_id = 0
            recorded_data = {}
            recorded_frames = []
        elif key == ord('w') or stop_flag:
            stop_flag = False
            print('End recording')
            recording = False
            gesture = input('Gesture name: ')
            if gesture in DYNAMIC_GESTURES:
                # save the recorded clip and skeleton file
                out = os.path.join(out_dir, gesture)
                if not os.path.exists(out):
                    os.makedirs(out)
                
                last_idx = len(os.listdir(out))
                os.mkdir(f'{out}/{last_idx}')
                # save skeleton json file
                with open(f'{out}/{last_idx}/data.json', 'w') as f:
                    json.dump(recorded_data, f)
                # save clip
                vid = cv2.VideoWriter(f'{out}/{last_idx}/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (400, 400), 0)
                print(recorded_frames[0].shape)
                for frame in recorded_frames:
                    vid.write(frame)
                vid.release()

            elif gesture in STATIC_GESTURES:
                out = os.path.join(out_dir, gesture)
                if not os.path.exists(out):
                    os.makedirs(out)
                
                last_idx = len(os.listdir(out))
                os.mkdir(f'{out}/{last_idx}')
                # save data file and images
                saved_data = {}
                for frame_id in recorded_data:
                    if frame_id % 10 == 0:
                        saved_data[frame_id] = recorded_data[frame_id]
                        # save image
                        cv2.imwrite(f'{out}/{last_idx}/sample_image_{frame_id}.png', recorded_frames[frame_id])

                with open(f'{out}/{last_idx}/data.json', 'w') as f:
                    json.dump(saved_data, f)


        elif key == ord('x'):
            break
        #display images
        
        # cv2.imwrite('Right.png', undistorted_right)


        # print("Frame id: %d, timestamp: %d, hands: %d, fingers: %d" % (
        #       frame.id, frame.timestamp, len(frame.hands), len(frame.fingers)))

        # Get hands
        # for hand in frame.hands:

        #     handType = "Left hand" if hand.is_left else "Right hand"

        #     print("  %s, id %d, position: %s" % (
        #         handType, hand.id, hand.palm_position))

        #     # Get the hand's normal vector and direction
        #     normal = hand.palm_normal
        #     direction = hand.direction

        #     # Calculate the hand's pitch, roll, and yaw angles
        #     print("  pitch: %f degrees, roll: %f degrees, yaw: %f degrees" % (
        #         direction.pitch * Leap.RAD_TO_DEG,
        #         normal.roll * Leap.RAD_TO_DEG,
        #         direction.yaw * Leap.RAD_TO_DEG))

        #     # Get arm bone
        #     arm = hand.arm
        #     print("  Arm direction: %s, wrist position: %s, elbow position: %s" % (
        #         arm.direction,
        #         arm.wrist_position,
        #         arm.elbow_position))

        #     # Get fingers
        #     for finger in hand.fingers:

        #         print("    %s finger, id: %d, length: %fmm, width: %fmm" % (
        #             self.finger_names[finger.type],
        #             finger.id,
        #             finger.length,
        #             finger.width))

        #         # Get bones
        #         for b in range(0, 4):
        #             bone = finger.bone(b)
        #             print("      Bone: %s, start: %s, end: %s, direction: %s" % (
        #                 self.bone_names[bone.type],
        #                 bone.prev_joint,
        #                 bone.next_joint,
        #                 bone.direction))

        # if not frame.hands.is_empty:
        #     print("")


def main():
    # Create a sample listener and controller
    # listener = SampleListener()
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

    # Have the sample listener receive events from the controller
    # controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        run(controller)
    except KeyboardInterrupt:
        sys.exit(0)
    # finally:
    #     controller.remove_listener(listener)

if __name__ == "__main__":
    main()