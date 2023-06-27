import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import LocalOutlierFactor as LOF
from pprint import pprint
from uncertainties import unumpy

from scc_lib import poly_fitting
from thermal_expansion import einstein_model
from read_yaml import construct_matrix, projection
from sympy import symbols, lambdify, Matrix, sign
from uncertainties import unumpy, umath


def mode_projection(motion_a, motion_b):
    return np.array(motion_a).T.dot(np.array(motion_b)).trace()


def mode_list_projection(motion_list_a, motion_list_b):
    # a is target, b is standard
    number_of_mode = len(motion_list_a)
    assert number_of_mode == len(motion_list_b), \
        f"the length of a and b should be the same, but {number_of_mode} and {len(motion_list_b)} are given."

    # the overlap are defined as <a_i|b_j>, which satisfies
    # <a_i|a_i> == 1, <a_i|a_j> == 0, <a_i|b_j> == <b_j|a_i>
    # \sigma_j |<a_i|b_j>|^2 == 1
    # sum_projection_square should be 1
    threshold = (1 / number_of_mode) ** .5
    projection_a_dict = {}
    for mode_index_a in range(number_of_mode):
        sum_projection_square = 0
        projection_b_dict = {}

        # mode_index_b start from mode_index_a and mode_index_a +-1, +-2, ...
        for mode_index_b in np.abs(np.arange(number_of_mode) - mode_index_a).argsort():
            projection = mode_projection(motion_list_a[mode_index_a], motion_list_b[mode_index_b])
            projection_b_dict[mode_index_b] = projection
            sum_projection_square += projection ** 2
            if sum_projection_square > 0.99:
                break
        else:
            # sum_projection_square < 0.99 eventually
            print("warning: orthogonality and normalization", mode_index_a, sum_projection_square)

        del_index_list = []
        for mode_index_b in projection_b_dict:
            if abs(projection_b_dict[mode_index_b]) < threshold:
                del_index_list.append(mode_index_b)

        for mode_index_b in del_index_list:
            del projection_b_dict[mode_index_b]

        projection_a_dict[mode_index_a] = projection_b_dict

    return projection_a_dict


def get_frequency_from_phonon_projection(projection_dict, standard_frequency):
    number_of_mode = len(projection_dict)
    assert number_of_mode == len(standard_frequency), \
        "the length of overlap_dict and standard_frequency should be the same, " \
        f"but {number_of_mode} and {len(standard_frequency)} are given."
    return [sum([standard_frequency[mode_index_b] * projection_dict[mode_index_a][mode_index_b] ** 2
                 for mode_index_b in projection_dict[mode_index_a]])
            for mode_index_a in range(number_of_mode)]


def get_shifted_frequency(coef_list, scale):
    if type(scale) in [np.ndarray, list]:
        scale = np.array(scale)
        return np.concatenate((scale[..., np.newaxis], np.ones_like(scale)[..., np.newaxis]), axis=-1).dot(coef_list)
    else:
        return np.array([scale, 1]).dot(coef_list)


if __name__ == "__main__":
    work_dir = r"C:\Users\dugue\OneDrive\spin-spin\444_520_PS"
    scale_list = [0.99, 1.0, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03]
    standard = 1.0
    prefix = "611"
    # -----------------------------------------------------
    # projection test

    # _, _, motion_a = json.load(open(work_dir + f"/mode_1.0.json", "r"))
    # _, _, motion_b = json.load(open(work_dir + f"/mode_1.0_400.json", "r"))
    # projection_dict = mode_list_projection(motion_a, motion_b)
    # ----------------------------------------------------------
    # phonon shift

    # frequency_dict = {}
    # motion_dict = {}
    # for scale in scale_list:
    #     _, frequency, motion = json.load(open(work_dir + f"/mode_{scale}.json", "r"))
    #     frequency_dict[scale] = frequency
    #     motion_dict[scale] = motion
    #
    # shifted_frequency_dict = {}
    # for scale in scale_list:
    #     if scale == standard:
    #         shifted_frequency_dict[scale] = frequency_dict[scale]
    #     else:
    #         projection_dict = mode_list_projection(motion_dict[standard], motion_dict[scale])
    #         json.dump(projection_dict, open(work_dir + f"/projection_{scale}_{standard}.json", "w"), indent=4)
    #         shifted_frequency_dict[scale] = get_frequency_from_phonon_projection(projection_dict, frequency_dict[scale])
    # json.dump(shifted_frequency_dict, open(work_dir + f"/shifted_frequency_{standard}.json", "w"))
    # -----------------------------------------------------------------------
    # phonon shift calculated with normalized atomic motions at V_cls

    shifted_frequency_dict = {}
    # TODO: error to deal with (fitted_frequency_error_list)
    for scale in scale_list:
        fitted_frequency_list, \
        fitted_frequency_error_list = json.load(open(work_dir + f"/fitted_frequency_{prefix}_{scale}.json", "r"))
        shifted_frequency_dict[scale] = fitted_frequency_list

    scale_list = np.array(list(map(float, shifted_frequency_dict.keys()))) - 1

    frequency_list = np.array(list(shifted_frequency_dict.values()))
    argsort = scale_list.argsort()
    scale_list = scale_list[argsort]
    frequency_list = frequency_list[argsort].T

    _coef_list = []
    _error_list = []
    for phonon_under_scale in frequency_list:
        coef, error, outer = poly_fitting(scale_list, phonon_under_scale, 1)
        _coef_list.append(coef.tolist())
        _error_list.append(error.tolist())
    json.dump([_coef_list, _error_list],
              open(work_dir + f"/shifted_frequency_fitting_with_const_motion_{prefix}.json", "w"))
    # ---------------------------------------------------------
    # fitting phonon frequencies

    # shifted_frequency_dict = json.load(open(work_dir + f"/shifted_frequency_{standard}.json", "r"))
    # scale_list = np.array(list(map(float, shifted_frequency_dict.keys()))) - 1
    #
    # frequency_list = np.array(list(shifted_frequency_dict.values()))
    # argsort = scale_list.argsort()
    # scale_list = scale_list[argsort]
    # frequency_list = frequency_list[argsort].T
    #
    # _coef_list = []
    # _error_list = []
    # for phonon_under_scale in frequency_list:
    #     coef, error, outer = poly_fitting(scale_list, phonon_under_scale, 1)
    #     _coef_list.append(coef.tolist())
    #     _error_list.append(error.tolist())
    # json.dump([_coef_list, _error_list], open(work_dir + "/shifted_frequency_fitting.json", "w"))
    # ----------------------------------------------------------------------------------------
    # plot diagram

    # scale_sample = np.linspace(scale_list[0], scale_list[-1], 100)
    # coef_list, error_list = json.load(open(work_dir + "/shifted_frequency_fitting.json", "r"))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for mode_index in range(3, 4):
    #     ax.scatter(scale_list, frequency_list[mode_index] / np.poly1d(coef_list[mode_index])(0) - 1, marker="x")
    #     ax.plot(scale_sample, np.poly1d(coef_list[mode_index])(scale_sample) / np.poly1d(coef_list[mode_index])(0) - 1)
    # ax.set_xlabel(r"$\frac{\delta L}{L_{\rm cls}}$")
    # ax.set_ylabel("relative variation of frequency")
    # plt.show()
    # ---------------------------------------------------------------
    # print(get_shifted_frequency(work_dir + "/shifted_frequency_fitting.json", 0)[3:10])
    # print(get_shifted_frequency(work_dir + "/shifted_frequency_fitting_with_const_motion.json", 0).shape)
