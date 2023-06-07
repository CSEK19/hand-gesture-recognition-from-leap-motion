import math
import json
import numpy as np

sensor_coordinate = (0,0,0)

class Plane3D:
  def __init__(self, normal_vector):
    self.normal_vector = normal_vector

def projection(vector, plane):
  normal = plane.normal_vector
  return vector - np.dot(vector, normal)/np.dot(normal,normal) * normal

def angle_between_two_vectors(v1, v2):
  unit_v1 = v1 / np.linalg.norm(v1)
  unit_v2 = v2 / np.linalg.norm(v2)
  dot_product = np.dot(unit_v1, unit_v2)
  angle = np.arccos(dot_product)
  return angle

def angle_between_vector_and_plane(vector, plane):
  normal = plane.normal_vector
  # Calculate the dot product of the vector and the plane's normal vector
  dot_product = np.dot(vector, normal)

  # Calculate the magnitude of the vector
  vector_norm = np.linalg(vector)

  # Calculate the magnitude of the plane's normal vector
  normal_norm = np.linalg(normal)

  # Calculate the angle between the vector and the plane
  cos_theta = dot_product / (vector_norm * normal_norm)
  theta = math.acos(cos_theta)

  return math.pi/2 - theta


def angle_pitch(vector):
  # pitch: angle between the negative z axis and the projection of hand direction on Oyz plane
  plane_oyz = Plane3D(np.array([1,0,0]))
  v_oz = np.array([0,0,-1])
  vector_p = projection(vector, plane_oyz)
  
  return angle_between_two_vectors(vector_p, v_oz)

def angle_yaw(vector):
  # yaw: angle between the negative z axis and the projection of hand direction on Oxz plane
  plane_oxz = Plane3D(np.array([0,-1,0]))
  v_oz = np.array([0,0,-1])
  vector_p = projection(vector, plane_oxz)

  return angle_between_two_vectors(vector_p, v_oz)

def angle_roll(vector):
  # roll: angle between the y axis and the projection of hand normal on Oxy plane
  plane_oxy = Plane3D(np.array([0,0,1]))
  v_oy = np.array([0,-1,0])
  vector_p = projection(vector, plane_oxy)

  return angle_between_two_vectors(vector_p, v_oy)

def to_degree(angle):
  return angle * 180 / math.pi

with open('D:\\bachkhoa\\bachkhoa222\\Luan van\\LeapMotionHGR\\data_collection\\LeapDatav4\\Phat\\left\\5\\data.json', 'r') as f:
  data = json.load(f)

for x in data:
  hands = data[x]["hands"]
  if len(hands) > 0:
    hand = hands[0]
    thumb = np.array(hand['thumb']).reshape(-1,3)
    index = np.array(hand['index']).reshape(-1,3)
    middle = np.array(hand['middle']).reshape(-1,3)
    ring = np.array(hand['ring']).reshape(-1,3)
    pinky = np.array(hand['pinky']).reshape(-1,3)

    direction = np.array(hand['direction'])
    normal = np.array(hand['normal'])
    palm = np.array(hand['palm'])
    wrist = np.array(hand['wrist'])

    sum_finger = (index[4] + middle[4] + ring[4])/3
    calculated_direction = np.array(middle[-1] - palm)

    calculated_angles = to_degree(angle_yaw(calculated_direction)), to_degree(angle_pitch(calculated_direction)), to_degree(angle_roll(normal))
    device_angles = to_degree(hand['yaw']), to_degree(hand['pitch']), to_degree(hand['roll'])
    print(calculated_angles, device_angles)
    # print(angle_between_two_vectors(direction, calculated_direction) * 180 / math.pi)
    # print(hand['middle'])

# print(get_angle(53.734962463378906, 193.784912109375, -2.4996438026428223, 31.290136337280273,180.6112060546875,44.09855651855469,))  