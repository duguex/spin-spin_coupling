# Import math module for trigonometric functions
import numpy as np
from scipy.spatial.transform import Rotation as R


def rot_stress_tensor(vector):
    stress_tensor = np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]])
    vector = vector / np.linalg.norm(vector)
    # x rot
    angle_x = np.arccos(np.dot(vector, [0, 0, 1]))
    # z rot
    vector[-1] = 0
    vector = vector / np.linalg.norm(vector)
    angle_z = np.arccos(np.dot(vector, [0, 1, 0]))

    # Initialize the rotation from the rotation vector
    # rot = R.from_rotvec(rotvec).as_matrix()
    rot = R.from_euler('x', angle_x).as_matrix().dot(R.from_euler('z', angle_z).as_matrix())

    return rot.T.dot(stress_tensor).dot(rot)


if __name__ == "__main__":
    print(rot_stress_tensor([1, 1, 1]))
