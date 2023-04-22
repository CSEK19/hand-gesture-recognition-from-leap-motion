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
    pyautogui.press('left')
  elif gesture == "move right":
    pyautogui.press('right')
  elif gesture == "close fist":
    pyautogui.press("up")
  elif gesture == "rotate":
    pyautogui.press("z")

def run():
  # init leap controller
  controller = Leap.Controller()
  controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
  
  # init hand gesture recognizer and visualizer
  vis = Visualizer()
  static_classifier = StaticHandPoseClassifier(static_model_weight, scaler_weight)

  frame_count = 0
  elapse = 0
  elapse_recognition = 0

  while True:
    prev_time = time.perf_counter()
    frame = controller.frame()
    image = frame.images[0]

    if image.is_valid:
      display = None
      if not frame.hands.is_empty:
        prev_time_recognition = time.perf_counter()
        frame_count += 1
        # make detection
        hand_features = extract_feature(frame)
        pose, score = static_classifier.predict_proba(hand_features)
        # if score >= 0.0:
        display = f"{pose}:{score}"

        elapse += (time.perf_counter() - prev_time)
        elapse_recognition += (time.perf_counter() - prev_time_recognition)
        if frame_count % 1000 == 0:
          print("Elapse(ms) per frame: ", elapse/frame_count * 1000)
          print("Elapse(ms) of recognition per frame: ", elapse_recognition / frame_count * 1000)

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

