# Import math module for trigonometric functions
import numpy as np
from scipy.spatial.transform import Rotation as R


def rotate_stress_tensor(stress_tensor, origin_direction, target_direction):
    _origin_direction = np.array(origin_direction)
    _origin_direction = _origin_direction / np.linalg.norm(_origin_direction)

    _target_direction = np.array(target_direction)
    _target_direction = _target_direction / np.linalg.norm(_target_direction)

    axis = np.cross(_target_direction, _origin_direction)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(_target_direction, _origin_direction))
    rot = R.from_rotvec(-axis * angle).as_matrix()

    # # x rot
    # angle_x = np.arccos(np.dot(_target_direction, _origin_direction))
    # # z rot
    # _target_direction[-1] = 0
    # _target_direction = _target_direction / np.linalg.norm(_target_direction)
    # angle_z = np.arccos(np.dot(_target_direction, [0, 1, 0]))
    #
    # # Initialize the rotation from the rotation vector
    # rot = R.from_euler('z', -angle_z).as_matrix().dot(R.from_euler('x', -angle_x).as_matrix())

    # rotate the origin z axis to check the matrix
    # print(rot.dot(np.array(_origin_direction)))
    return rot.dot(stress_tensor).dot(rot.T)


if __name__ == "__main__":
    print(rotate_stress_tensor(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]), [0, 0, 1], [1, 1, 1]))
    # print(rot_stress_tensor(np.array([[0, 0, .5], [0, 0, 0], [.5, 0, 0]]), [1, 1, 1]))
