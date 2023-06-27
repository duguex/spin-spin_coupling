#!/usr/bin/env python
# coding: utf-8

import poscar
import sys
import numpy as np
import scc_lib
import os
import matplotlib.pyplot as plt
import random
import matplotlib

import json
from scipy import stats
import pprint
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator
import read_yaml
import json_tricks
from thermal_expansion import einstein_model

# matplotlib.rcParams["backend"] = "PDF"
# plt.rcParams["font.size"] = 15
dump_fig = True
dump_json = False
fig_type = "png"
work_dir = "E:/OneDrive/444"
# work_dir = "E:/OneDrive/1226_strain"
K = 8.6173e-05
C = 239.203931
thermal_expansion_model = einstein_model()
'''
kg to amu 6.0221E+26
hbar 1.05457266E-34 J·s
e 1.602176634E-19 C
eV to J 1.602176634E-19
m * freq**2 * x**2 = hbar * freq 在国际单位制下成立
对于本计算的单位制 amu eV A hbar=1
6.0221E+26 * (1.05457266E-34/1.602176634E-19)**2 * (1E+10)**2 * C**2= 1/1.602176634E-19 * C
C = 1/1.602176634E-19 / (6.0221E+26 * (1.05457266E-34/1.602176634E-19)**2 * (1E+10)**2)
  = 239.22657077980037
'''

atom_name = ["$^{14}N$", "$^{13}C(1)$", "$^{13}C(2)$", "$^{13}C(3)$", "$^{13}C(4)$", "$^{13}C(5)$"]

# 3\times 3\times 3
# atom_list = [1, 2, 92, 32, 57, 55]
# atom_list = [92]


# 4\times 4\times 4
atom_list = [1, 2, 153, 27, 232, 231]

"""
N 0
C1 1,2,3
C2 152,155,279,293,408,420
C3 26,29,41
C4 231,234,346,361,473,485
C5 230,345,469
"""

v_parallel = np.array([1, 1, 1])
v_perpendicular = np.array([1, -1, 0])


# atom_list = [1]


# atom_list = [1, 3, 421, 42, 362, 231]


# atom_list = [153]

# symmetry test
# atom_list = [1,
#              2, 3, 4,
#              153, 156, 280, 294, 409, 421,
#              27, 30, 42,
#              232, 235, 347, 362, 474, 486,
#              231, 346, 470]





def RotationMatrix(t1, t2, t3):
    return np.array([[np.cos(t1) * np.cos(t3) - np.cos(t2) * np.sin(t1) * np.sin(t3),
                      np.cos(t3) * np.sin(t1) + np.cos(t1) * np.cos(t2) * np.sin(t3),
                      np.sin(t2) * np.sin(t3)],
                     [-np.cos(t2) * np.cos(t3) * np.sin(t1) - np.cos(t1) * np.sin(t3),
                      np.cos(t1) * np.cos(t2) * np.cos(t3) - np.sin(t1) * np.sin(t3),
                      np.sin(t2) * np.cos(t3)],
                     [np.sin(t1) * np.sin(t2), -np.cos(t1) * np.sin(t2), np.cos(t2)]])


# 111 to 001
# rot = (0, 0.9553166181245092, 0.7853981633974483)
# rot = (0, 0, 0)
# rot_mat = RotationMatrix(*rot)
# vib0 = json.load(open(work_dir + "/xx+yy+zz.json", "r"))
# for k in vib0:
#     if k["path"]=="xx+yy+zz/0.0/OUTCAR":
#         for l in k["hyperfine_tensor"]:
#             print(l)
#             a_rot = rot_mat.T.dot(np.array(k["hyperfine_tensor"][l]).dot(rot_mat))
#             array_string = ""
#             for i in a_rot:
#                 for j in i:
#                     array_string += str(round(j, 2)) + "&"
#                 array_string = array_string[:-1] + r"\\" + "\n"
#
#             print(array_string)

if __name__ == '__main__':
    # expand flag
    including_expand = False
    including_vib = True
    including_strain = False

    '''
    A matrix for 14N at 300 K

    first order derivative
    array([[152.95169475,  20.28044501,  20.28891591],
       [ 20.28044501, 152.90507988,  20.27496795],
       [ 20.28891591,  20.27496795, 152.89555092]])

    (array([193.4803414 , 132.64909562, 132.62288853]),
    array([[-0.57773674, -0.77698971, -0.25001451],
        [-0.57716192,  0.60547805, -0.54797851],
        [-0.57715196,  0.17228847,  0.79825578]]))

    second order derivative
    array([[5.74772681e-07, 6.78599706e-08, 6.78970599e-08],
       [6.78599706e-08, 5.74888976e-07, 6.79484675e-08],
       [6.78970599e-08, 6.79484675e-08, 5.74939316e-07]])

    (array([7.10670711e-07, 5.06958211e-07, 5.06972051e-07]),
     array([[ 0.57695073, -0.59978221, -0.55442688],
            [ 0.57742604, -0.18056901,  0.79622484],
            [ 0.57767381,  0.77952302, -0.24215044]]))

    '''
    if including_expand:
        # expand part
        # 3\times 3\times 3
        # expand_dict = {1: -0.14890109890108946,
        #                2: -49.128574909854834,
        #                92: 2.484024170945024,
        #                32: 3.379439422635399,
        #                57: -3.620531879907429,
        #                55: -1.3553078631494433, }
        # 4\times 4\times 4
        # expand_dict = {1: -0.09733823529411631,
        #                2: -36.02762342878353,
        #                153: 1.7437037436486906,
        #                27: 2.1418375610626548,
        #                232: -2.4736870204068118,
        #                231: -0.6619703385867863, }

        # A_expand_data = json.load(open(work_dir + "/expand.json", "r"))
        A_expand_data = json.load(open(work_dir + "/xx+yy+zz.json", "r"))
        # for obj in A_expand_data:
        #     for tag in ["hyperfine_p", "hyperfine_eig", "hyperfine_v"]:
        #         del obj[tag]
        #     obj["hyperfine_para"] = {atom: read_yaml.projection(obj['hyperfine_tensor'][atom], v_parallel) for atom in
        #                                  obj['hyperfine_tensor']}
        #     obj["hyperfine_perp"] = {atom: read_yaml.projection(obj['hyperfine_tensor'][atom], v_perpendicular) for atom in
        #                                  obj['hyperfine_tensor']}
        # json.dump(A_expand_data,open(work_dir + "/xx+yy+zz2.json", "w"),indent=4)

        # expand for 400 PE
        # expand_520_ps for 520 PS
        A_expand_data.sort(key=lambda obj: obj["length_of_vectors"])

        length_of_vectors = [obj['length_of_vectors'] for obj in A_expand_data]
        # length_of_vectors_0 = length_of_vectors[0]
        length_of_vectors_0 = length_of_vectors[len(length_of_vectors) // 2]
        length_of_vectors = np.array(length_of_vectors) / length_of_vectors_0 - 1
        # energy = np.array([obj['energy'] for obj in A_expand_data])
        A_expand_in_atom = {atom: {"fit_mat": np.zeros(9).reshape(3, 3),
                                   "fit0_mat": np.zeros(9).reshape(3, 3),
                                   "error_mat": np.zeros(9).reshape(3, 3),
                                   "fit_para": 0,
                                   "fit0_para": 0,
                                   "error_para": 0,
                                   "fit_perp": 0,
                                   "fit0_perp": 0,
                                   "error_perp": 0, }
                            for atom in atom_list}
        # fermi_expand_in_atom = {atom: {"fit": [], "error": []} for atom in atom_list}
        # pprint.pprint(A_expand_data[0]['hyperfine'])
        for atom in atom_list:
            # fit_res = fitting(length_of_vectors, [obj['hyperfine'][str(atom)] for obj in A_expand_data], 1,
            #                   check=True)
            # fit_res = fitting(length_of_vectors, [obj['hyperfine_v'][str(atom)] for obj in A_expand_data], 1,
            #                   check=True)
            for col in range(3):
                for row in range(3):
                    fit_res = poly_fitting(length_of_vectors,
                                           [obj['hyperfine_tensor'][str(atom)][col][row] for obj in A_expand_data], 1,
                                           check=True)
                    A_expand_in_atom[atom]["fit_mat"][col][row] = fit_res["fit"][0]
                    A_expand_in_atom[atom]["fit0_mat"][col][row] = fit_res["fit"][1]
                    A_expand_in_atom[atom]["error_mat"][col][row] = fit_res["error"][0]

            fit_res = poly_fitting(length_of_vectors, [obj['hyperfine_para'][str(atom)] for obj in A_expand_data], 1,
                                   check=True)
            A_expand_in_atom[atom]["fit_para"] = fit_res["fit"][0]
            A_expand_in_atom[atom]["fit0_para"] = fit_res["fit"][1]
            A_expand_in_atom[atom]["error_para"] = fit_res["error"][0]

            fit_res = poly_fitting(length_of_vectors, [obj['hyperfine_perp'][str(atom)] for obj in A_expand_data], 1,
                                   check=True)
            A_expand_in_atom[atom]["fit_perp"] = fit_res["fit"][0]
            A_expand_in_atom[atom]["fit0_perp"] = fit_res["fit"][1]
            A_expand_in_atom[atom]["error_perp"] = fit_res["error"][0]

            # plot=f"expand_fit_check_{atom}"

            # fit_res = fitting(length_of_vectors, [obj['fermi_contact'][str(atom)] for obj in A_expand_data], 1,
            #                   check=True)
            # # plot=f"expand_fit_check_{atom}"
            # fermi_expand_in_atom[atom]["fit"] = fit_res["fit"][0]
            # fermi_expand_in_atom[atom]["error"] = fit_res["error"][0]

            # paper plot
            if True:
                x = np.linspace(min(length_of_vectors), max(length_of_vectors), 100)
                fig = plt.figure()
                for col in range(3):
                    for row in range(3):
                        index = 331 + col * 3 + row
                        ax = fig.add_subplot(index)
                        ax.scatter(length_of_vectors * 100,
                                   [obj['hyperfine_tensor'][str(atom)][col][row] for obj in A_expand_data])
                        ax.plot(x * 100,
                                A_expand_in_atom[atom]["fit_mat"][col][row] * x +
                                A_expand_in_atom[atom]["fit0_mat"][col][row], color="red")
                        ax.xaxis.set_major_locator(MaxNLocator(5))
                        ax.yaxis.set_major_locator(MaxNLocator(5))

                fig.tight_layout()
                if dump_fig:
                    plt.savefig(f"lattice_fit_mat_{atom}.{fig_type}", dpi=600, format=fig_type)
                plt.close("all")

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(length_of_vectors * 100,
                           [obj['hyperfine_para'][str(atom)] for obj in A_expand_data])
                ax.plot(x * 100,
                        A_expand_in_atom[atom]["fit_para"] * x + A_expand_in_atom[atom]["fit0_para"], color="red")
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_major_locator(MaxNLocator(5))

                fig.tight_layout()
                if dump_fig:
                    plt.savefig(f"lattice_fit_para_{atom}.{fig_type}", dpi=600, format=fig_type)
                plt.close("all")

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(length_of_vectors * 100,
                           [obj['hyperfine_perp'][str(atom)] for obj in A_expand_data])
                ax.plot(x * 100,
                        A_expand_in_atom[atom]["fit_perp"] * x + A_expand_in_atom[atom]["fit0_perp"], color="red")
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_major_locator(MaxNLocator(5))

                fig.tight_layout()
                if dump_fig:
                    plt.savefig(f"lattice_fit_perp_{atom}.{fig_type}", dpi=600, format=fig_type)
                plt.close("all")

        pprint.pprint(A_expand_in_atom)


    if including_strain:
        fit_res_colle = {}
        for json_filename in os.listdir(work_dir):
            if json_filename[-5:] == ".json" and "dontread" not in json_filename:
                print(json_filename[:-5])
                fit_res_colle[json_filename[:-5]] = {}
                data = json.load(open(f"{work_dir}/{json_filename}", "r"))
                for obj in data:
                    obj["delta"] = float(obj["path"].split("/")[1])
                data.sort(key=lambda obj: obj["delta"])
                delta_list = np.array([obj['delta'] for obj in data])
                x = np.linspace(delta_list[0], delta_list[-1], 100)
                zero_index = np.where(delta_list == 0)[0][0]

                energy = np.array([obj['energy'] for obj in data]) - data[zero_index]['energy']
                hyperfine = {atom: np.array([obj['hyperfine_v'][str(atom)] for obj in data]) for atom in atom_list}

                fig = plt.figure()

                # A for 6 nuclei
                ax_list = [fig.add_subplot(panel) for panel in range(237)[231:]]
                for i in range(6):
                    fit_res = poly_fitting(delta_list, hyperfine[atom_list[i]], 1, check=True)
                    fit_res_colle[json_filename[:-5]][atom_name[i]] = fit_res
                    fit, error = list(zip(*fit_res.values()))[0]
                    print(atom_name[i], fit, error)
                    # for fit, error in zip(*fit_res.values()):
                    #     print(f"{fit} ± {error} {abs(error/fit)}")

                    ax_list[i].scatter(delta_list, hyperfine[atom_list[i]])
                    if abs(fit) < error * 3:
                        pass
                    else:
                        ax_list[i].plot(x, np.poly1d(fit_res["fit"])(x), "r")
                    # ax_list[i].set_title(
                    #     f"{atom_name[i]} {round(fit_res['fit'][1], 2)}±{round(fit_res['error'][1], 2)}")
                    ax_list[i].set_title(atom_name[i])
                    ax_list[i].yaxis.set_major_locator(MaxNLocator(6))

                # energy
                # ax = fig.add_subplot(111)
                # fit_res = fitting(delta_list, energy, 2)
                # fit_res_colle[json_filename[:-5]]["energy"] = fit_res
                # ax.scatter(delta_list, energy)
                # ax.plot(x, np.poly1d(fit_res["fit"])(x), "r")
                # # ax.set_title(f"energy {round(fit_res['fit'][0], 2)}±{round(fit_res['error'][0], 2)}")
                # for fit, error in zip(*fit_res.values()):
                #     print(f"{fit} ± {error} {abs(error / fit)}")

                fig.tight_layout()
                plt.savefig(f"{work_dir}/strain_{'.'.join(json_filename.split('.')[:-1])}.png", dpi=300)
                plt.close("all")
        json.dump(fit_res_colle, open(f"{work_dir}/dontread_fit_res_colle.json", "w"), indent=4)
