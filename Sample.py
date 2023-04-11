################################################################################
# Copyright (C) 2012-2018 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
################################################################################

import Leap, sys, _thread, time

import cv2, Leap, math, ctypes
import numpy as np
import matplotlib.pyplot as plt

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

    #wrap image data in numpy array
    i_address = int(image.data_pointer)
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


class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']

    def on_init(self, controller):
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()
        if not frame.hands.is_empty:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            # image = controller.images[0]
            # maps_initialized = False
            # # print(image.is_valid)
            # if image.is_valid:
            #     print('haha')
            #     if not maps_initialized:
            #         left_coordinates, left_coefficients = convert_distortion_maps(controller.images[0])
            #         right_coordinates, right_coefficients = convert_distortion_maps(controller.images[1])
            #         maps_initialized = True

            #     undistorted_left = undistort(image, left_coordinates, left_coefficients, 400, 400)
            #     undistorted_right = undistort(image, right_coordinates, right_coefficients, 400, 400)
            #     print('??')
            #     #display images
            #     cv2.imwrite('Left.png', undistorted_left)
            #     cv2.imwrite('Right.png', undistorted_right)
            # return

            print("Frame id: %d, timestamp: %d, hands: %d, fingers: %d" % (
                frame.id, frame.timestamp, len(frame.hands), len(frame.fingers)))

            # Get hands
            for hand in frame.hands:

                handType = "Left hand" if hand.is_left else "Right hand"
                '''
                print("  %s, id %d, position: %s" % (
                    handType, hand.id, hand.palm_position))
                '''
                # Get the hand's normal vector and direction
                normal = hand.palm_normal
                direction = hand.direction

                # Calculate the hand's pitch, roll, and yaw angles
                print("  pitch: %f degrees, roll: %f degrees, yaw: %f degrees" % (
                    direction.pitch * Leap.RAD_TO_DEG,
                    normal.roll * Leap.RAD_TO_DEG,
                    direction.yaw * Leap.RAD_TO_DEG))
                
                ax.scatter(hand.arm.wrist_position.x, hand.arm.wrist_position.z, hand.arm.wrist_position.y, marker='o')

                ax.scatter(0, 0, 0, marker='o')
                
                # Get arm bone
                arm = hand.arm
                # print("  Arm direction: %s, wrist position: %s, elbow position: %s" % (
                #     arm.direction,
                #     arm.wrist_position,
                #     arm.elbow_position))

                # Get fingers
                for finger in hand.fingers:

                    # print("    %s finger, id: %d, length: %fmm, width: %fmm" % (
                    #     self.finger_names[finger.type],
                    #     finger.id,
                    #     finger.length,
                    #     finger.width))

                    # Get bones
                    prev_joint = (hand.arm.wrist_position.x, hand.arm.wrist_position.y, hand.arm.wrist_position.z)
                    for b in range(0, 4):
                        bone = finger.bone(b)
                        # print("      Bone: %s, start: %s, end: %s, direction: %s" % (
                        #     self.bone_names[bone.type],
                        #     bone.prev_joint,
                        #     bone.next_joint,
                        #     bone.direction))
                        joint = (bone.prev_joint + bone.next_joint) / 2
                        ax.plot([prev_joint[0], bone.next_joint.x], [prev_joint[2], bone.next_joint.z], [prev_joint[1], bone.next_joint.y])
                        prev_joint = (bone.next_joint.x, bone.next_joint.y, bone.next_joint.z)
                plt.show()
        if not frame.hands.is_empty:
            print("")

def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
    # controller.set_policy_flags(Leap.Controller.POLICY_RAW_IMAGES)
    # controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
