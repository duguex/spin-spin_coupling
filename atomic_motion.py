'''

given nuclues and mode, analyse the atomic factor

'''
import poscar
import os
import numpy as np
import pprint
import json_tricks

grd = poscar.Poscar("POSCAR").c2d()


def mode_json(atom, mode):
    amp_motion = []
    for dir in os.listdir():
        if os.path.isdir(dir) and "p" + str(mode) == dir.split("_")[0]:
            motion = poscar.Poscar(dir + "/POSCAR").c2d().position[atom - 1] - grd.position[atom - 1]
            modulus = np.linalg.norm(motion, 2)
            motion /= modulus
            amp_motion.append([float(dir.split("_")[1]), modulus, motion])
    amp_motion.sort(key=lambda x: x[0])

    pprint.pprint(amp_motion)

if __name__=="__main__":
    mode_json(1, 16)
