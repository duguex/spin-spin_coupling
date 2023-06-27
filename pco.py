#!/usr/bin/env python
# coding: utf-8

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import LocalOutlierFactor as LOF

from thermal_expansion import einstein_model
from read_yaml import construct_matrix, projection


def light_outcar(outcar_json_path, atom_list):
    outcar_json = json.load(open(outcar_json_path, "r"))
    outcar_json_path = outcar_json_path.split("/")[-1]
    scale = float(outcar_json_path[:-5].split("_")[1])
    light_one = []
    for obj in outcar_json:
        # frozen phonon 'p_285_1.5/OUTCAR'
        # static '1.045/OUTCAR'
        if "_" in obj["path"]:
            mode, displace = obj["path"][:-7].split("_")[1:]
            mode = int(mode)
            displace = float(displace)
        else:
            mode = 0
            displace = 0

        # the keys of outcar_json are
        # ['path', 'energy', 'length_of_vectors', 'quadrupole_moment', 'fermi_contact', 'dipolar']

        energy = obj["energy"]
        quadrupole_moment = obj["quadrupole_moment"]
        fermi_contact = np.array(obj["fermi_contact"])[atom_list].tolist()
        dipolar = np.array(obj["dipolar"])[atom_list].tolist()
        light_one.append([scale, mode, displace, energy, quadrupole_moment] + fermi_contact + sum(dipolar, []))
    return light_one


def poly_fitting(x, y, fitting_order):
    _x = np.array(x).copy()
    _y = np.array(y).copy()

    # clf = LOF(n_neighbors=2)
    # predict = clf.fit_predict(np.concatenate((_x[np.newaxis, ...], _y[np.newaxis, ...]), axis=0).T)
    # outer = np.where(predict == -1)
    # coef, cov = np.polyfit(np.delete(_x, outer), np.delete(_y, outer), fitting_order, cov=True)

    # skip LOF
    coef, cov = np.polyfit(_x, _y, fitting_order, cov=True)
    outer = [np.array([])]
    # poly_plot(x, y, coef, outer)

    return [coef, np.sqrt(cov.diagonal()), outer]


def value_fitting(displace, value, mode, _frequency, energy_index):
    assert len(displace) == len(value) > 3, "The size of x and y should be the same and larger than 3 to fit, " \
                                            f"but {len(displace)} and {len(value)} are given."

    # fitting displacement and values for specific mode
    _coef_list = []
    _error_list = []

    displace = np.array(displace)
    argsort = displace.argsort()
    displace = displace[argsort]
    value = list(np.array(value)[argsort].T)
    for value_index in range(len(value)):
        if value_index == energy_index:
            # coef, cov = curve_fit(lambda x, a, b: a * x ** 3 + b * x ** 2, displace, energy, bounds=([-1, 0], [1, 4]))
            # cov = np.sqrt(np.diag(cov))
            # coef, error, outer = poly_fit(displace, value[value_index], 3)
            coef, error, outer = poly_fitting(displace, value[value_index], 2)

            # mass check
            # a * x^2 + b * x + c
            # a==frequency ** 2 / 2 * C b==c==0

            theoretical_second_order_coef = _frequency ** 2 / 2 * C
            if abs(coef[0] - theoretical_second_order_coef) / theoretical_second_order_coef > .05:
                print(f"mass check failed for mode {mode}: "
                      f"mass check fitting: {coef}, "
                      f"harmonic: {theoretical_second_order_coef}")
                # plot
                mass_check(displace, value[value_index], coef, _frequency, sample=100)
        else:
            coef, error, outer = poly_fitting(displace, value[value_index], 2)
        if not outer[0].shape == (0,):
            print(f"outer(s) when fitting: mode = {mode}, displace = {displace[outer]}")
            # poly_plot(displace, value[value_index], coef, outer)

        _coef_list.append(coef.tolist())
        _error_list.append(error.tolist())
    return [_coef_list, _error_list]


def poly_plot(x, y, coef, outer):
    _x = x.copy()
    _y = y.copy()

    x_removed = _x[outer]
    y_removed = _y[outer]

    x_min = _x[0]
    x_max = _x[-1]

    _x = np.delete(_x, outer)
    _y = np.delete(_y, outer)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(_x, _y)
    x_sample = np.linspace(x_min, x_max, 100)
    ax.plot(x_sample, np.poly1d(coef)(x_sample))
    if not outer[0].shape == (0,):
        ax.scatter(x_removed, y_removed, marker="x", color="r")

    ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
    ax.tick_params(axis='both', which='major')
    ax.minorticks_on()
    ax.yaxis.set_major_locator(MaxNLocator(5))
    plt.tight_layout()
    # plt.savefig(f"{plot}.png", dpi=600, format="png")
    # plt.close("all")
    plt.show()


def mode_projection(motion_a, motion_b):
    return np.array(motion_a).T.dot(np.array(motion_b)).trace()


def mode_list_projection(motion_list_a, motion_list_b):
    # a is target, b is standard
    number_of_mode = len(motion_list_a)
    assert number_of_mode == len(motion_list_b), \
        f"the length of a and b should be the same, but {number_of_mode} and {len(motion_list_b)} are given."

    overlap_dict = {}
    for mode_index_a in range(number_of_mode):
        overlap = 0
        overlap_dict[mode_index_a] = {}
        mode_shift = 0
        while overlap < 0.97 and mode_shift < max(mode_index_a, number_of_mode - mode_index_a):
            for mode_index_b in [mode_index_a - mode_shift, mode_index_a + mode_shift] if mode_shift > 0 \
                    else [mode_index_a]:
                # whether mode_index_b is legal
                if number_of_mode > mode_index_b >= 0:
                    projection = mode_projection(motion_list_a[mode_index_a], motion_list_b[mode_index_b])
                    if abs(projection) > 0.001:
                        overlap_dict[mode_index_a][mode_index_b] = projection
                        overlap += projection ** 2
                # print(mode_index_a, mode_index_b, overlap)

            mode_shift += 1

    return overlap_dict


def get_frequency(overlap_dict, standard_frequency):
    number_of_mode = len(overlap_dict)
    assert number_of_mode == len(standard_frequency), \
        "the length of overlap_dict and standard_frequency should be the same, " \
        f"but {number_of_mode} and {len(standard_frequency)} are given."
    return [sum([standard_frequency[mode_index_b] * overlap_dict[mode_index_a][mode_index_b] ** 2
                 for mode_index_b in overlap_dict[mode_index_a]])
            for mode_index_a in range(number_of_mode)]


def mass_check(x, y, coef, frequency, sample=100):
    x_sample = np.linspace(x[0], x[-1], sample)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y - coef[-1])
    ax.plot(x_sample, np.poly1d(coef)(x_sample) - coef[-1], "r--", label="fitting")
    ax.plot(x_sample, (frequency * x_sample) ** 2 / 2 * C, label="harmonic")

    # # +声子波函数
    # number_of_phonon = 0
    #
    # interval = (x[-1] - x[0]) / sample
    # while True:
    #     phonon_energy = frequency * (number_of_phonon + 0.5)
    #     if phonon_energy > min(np.poly1d(coef)([x[0], x[-1]])):
    #         break
    #     else:
    #         ax.plot([x[0], x[-1]], [phonon_energy, phonon_energy], "--", linewidth=0.7)
    #         _x = x_sample * (frequency * C) ** 0.5
    #         phonon_wf = np.e ** (-_x ** 2 / 2) * \
    #                     np.polynomial.hermite.hermval(_x, np.append(np.zeros(number_of_phonon), 1))
    #         # 0.25 跟据显示效果可以调整
    #         ax.plot(x_sample, .25 * frequency / (sum(phonon_wf ** 2) * interval) ** 0.5 * phonon_wf +
    #                 frequency * (number_of_phonon + 0.5), linewidth=0.7)
    #         number_of_phonon += 1

    ax.set_xlabel(r'$X$ ($\rm \sqrt{amu}\cdot\AA$)')
    ax.set_ylabel("Energy (eV)")
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.minorticks_on()
    ax.yaxis.set_major_locator(MaxNLocator(5))
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"mass_check_{mode_index}.pdf", dpi=600, format="pdf")
    # plt.close("all")


def fitting_frozen_phonon_at_fixed_volume(value_list, key_list, scale, frequency, mode_start, mode_end):
    """
        某模式单个声子对应的温度 frequency[nmode] / np.log(2) / K K 和振幅 np.sqrt(2 / frequency[nmode] / C) amu^(1/2)A
        300K 对应的声子数 number_of_phonon_for_specfic_mode = 1 / (np.exp(frequency[nmode] / (K * 300)) - 1)
        和振幅 np.sqrt(2 * number_of_phonon_for_specfic_mode / frequency[nmode] / C) amu^(1/2)A
    """

    scale_index = key_list.index("scale")
    mode_index = key_list.index("mode")
    displace_index = key_list.index("displace")
    energy_index = key_list.index("energy")
    # keys are "scale", "mode", "displace"
    value_start_from = 3
    energy_index -= value_start_from
    number_of_value = len(key_list) - value_start_from

    frozen_phonon_list = list(filter(lambda x: x[scale_index] == scale, value_list))
    frozen_phonon_list.sort(key=lambda x: x[mode_index])
    static = frozen_phonon_list.pop(0)
    frozen_phonon_list = list(filter(lambda x: x[mode_index] in range(mode_start, mode_end), frozen_phonon_list))

    # for each mode there are 1 * energy, 1 * quadrupole_moment for 14N,
    # atom_list * fermi_contact and (atom_list * 6) * dipolar need to fit

    # skipping mode 0, 1, 2
    current_mode = mode_start

    _displace_list = []
    _value_list = []
    # fitting_list[mode] = [[coef, ...],[error, ...]] for each value
    coef_list = []
    error_list = []
    # _coef_list = [[coef, error], ...]

    for frozen_phonon_sample in frozen_phonon_list:
        if frozen_phonon_sample[mode_index] == current_mode:
            _displace_list.append(frozen_phonon_sample[displace_index])
            _value_list.append(frozen_phonon_sample[value_start_from:])
        else:
            # print(current_mode)
            assert static[displace_index] == 0, "the displace of static configure should be 0."
            # adding static sample whose displace == 0
            _displace_list.append(static[displace_index])
            _value_list.append(static[value_start_from:])
            _coef_list, _error_list = value_fitting(_displace_list, _value_list,
                                                    current_mode, frequency[current_mode],
                                                    energy_index)
            coef_list.append(_coef_list)
            error_list.append(_error_list)
            # start dealing with the next mode
            current_mode += 1
            _displace_list = [frozen_phonon_sample[displace_index]]
            _value_list = [frozen_phonon_sample[value_start_from:]]
    # for the last mode
    # print(current_mode)
    assert static[displace_index] == 0, "the displace of static configure should be 0."
    # adding static sample whose displace == 0
    _displace_list.append(static[displace_index])
    _value_list.append(static[value_start_from:])
    _coef_list, _error_list = value_fitting(_displace_list, _value_list,
                                            current_mode, frequency[current_mode],
                                            energy_index)
    coef_list.append(_coef_list)
    error_list.append(_error_list)

    return [coef_list, error_list]


# anharmonic is from anharmonic vibration ax3/<x> and linear x dependent A(x)=ax


def frozen_phonon_at_various_volume(coef_list, frequency):
    # 二阶导和二次项系数差2
    # 在 0 处的二阶导, assuming A = a * x ** 2, A''|0 = 2 * a
    # np.polyval(np.polyder(np.polyder(coef)), 0)
    # for specific displacement, 2 * a = (A[displacement] + A[-displacement] - 2 * A0) / displacement ** 2

    # t[289] == 300
    # t_sample.shape == (490,)
    t_sample = np.array(range(501)[11:])

    # coef_list.shape == (1530, 156, 3)
    second_order_coef = coef_list.transpose(2, 1, 0)[0]

    # second_order_coef.shape == (156, 1530)
    t_sample = np.array(range(501)[11:])

    # 1 / (exp(f / k / T) - 1)
    # number_of_phonon.shape == (490, 1530)
    number_of_phonon = 1 / (np.exp(np.outer(1 / t_sample, frequency[3:]) / K) - 1)

    # A * (n / f) / C
    # Ci = A / f / C
    # (156, 1530) @ (1530, 490) = (156, 490)
    delta_A = second_order_coef @ (number_of_phonon / frequency[3:]).T / C
    # when number_of_phonon == 0.5 gives zero point

    # A * (n + 1) * n / T / T / K / C
    # (156, 1530) * (490, 1530) * (490, 1530) / (490,) / (490,) / K / C
    dA_dT = second_order_coef @ ((number_of_phonon * (number_of_phonon + 1)).T / np.power(t_sample, 2)) / K / C


def get_shifted_frequency(scale):
    # phonon shift

    # scale_list = [0.99, 1.0, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03]
    #
    # frequency_dict = {}
    # motion_dict = {}
    # for scale in scale_list:
    #     _, frequency, motion = json.load(open(work_dir + f"/mode_{scale}.json", "r"))
    #     frequency_dict[scale] = frequency
    #     motion_dict[scale] = motion
    #
    # standard = 1.0
    # shifted_frequency_dict = {}
    # for scale in scale_list:
    #     if scale == standard:
    #         shifted_frequency_dict[scale] = frequency_dict[scale]
    #     else:
    #         overlap_dict = mode_list_projection(motion_dict[standard], motion_dict[scale])
    #         json.dump(overlap_dict, open(work_dir + f"/projection_{scale}_{standard}.json", "w"), indent=4)
    #         shifted_frequency_dict[scale] = get_frequency(overlap_dict, frequency_dict[scale])
    # json.dump(shifted_frequency_dict, open(work_dir + f"/shifted_frequency_{standard}.json", "w"))
    # ---------------------------------------------------------
    # standard = 1.0
    # shifted_frequency_dict = json.load(open(work_dir + f"/shifted_frequency_{standard}.json", "r"))
    # for _scale in [0.97, 0.98]:
    #     del shifted_frequency_dict[str(_scale)]
    # scale_list = np.array(list(map(float, shifted_frequency_dict.keys())))
    #
    # frequency_list = np.array(list(shifted_frequency_dict.values()))
    # argsort = scale_list.argsort()
    # scale_list = scale_list[argsort]
    # # 1533 * scale
    # frequency_list = frequency_list[argsort].T
    # _coef_list = []
    # for phonon_under_scale in frequency_list:
    #     coef, error, outer = poly_fitting(scale_list, phonon_under_scale, 1)
    #     _coef_list.append(coef.tolist())
    # json.dump(_coef_list, open(work_dir + "/shifted_frequency_fitting.json", "w"))

    # plot diagram
    # scale_sample = np.linspace(scale_list[0], scale_list[-1], 100)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for mode_index in range(3, 1500, 250):
    #     ax.scatter(scale_list, frequency_list[mode_index])
    #     ax.plot(scale_sample, np.poly1d(_coef_list[mode_index])(scale_sample))
    # ax.set_xlabel("Scale Factor")
    # ax.set_ylabel("Frequency (eV)")
    # plt.show()
    # --------------------------------------------------------------------------------------------
    _coef_list = json.load(open(work_dir + "/shifted_frequency_fitting.json", "r"))
    # _coef_list.shape = (1533, 2)
    _coef_list = np.array(_coef_list)
    return _coef_list.dot([scale, 1])


def find_V(eos, p, t):
    for state in eos:
        if state[0] == p and state[1] == t:
            _v = state[2]
            break
    else:
        print(f"p = {p}, T = {t} not found.")
        _v = None
    return _v


if __name__ == "__main__":
    # test for poly_fit
    # x = np.arange(5)
    # y = np.power(x, 2)
    # y[0] -= 10
    # coef, error, outer = poly_fit(x, y, 2)
    # poly_plot(x, y, coef, outer)
    # exit()
    # --------------------------------------------------------------------------

    # settings
    # matplotlib.rcParams["backend"] = "PDF"
    # plt.rcParams["font.size"] = 15
    dump_fig = True
    dump_json = False
    fig_type = "png"
    work_dir = r"C:\Users\dugue\OneDrive\spin-spin\444_520_PS"
    K = 8.6173e-05
    C = 239.203931
    thermal_expansion_model = einstein_model()

    '''
    kg to amu 6.0221E+26
    hbar 1.05457266E-34 J·s
    e 1.602176634E-19 C
    eV to J 1.602176634E-19
    m * frequency**2 * x**2 = hbar * frequency 在国际单位制下成立
    对于本计算的单位制 amu eV A hbar=1
    6.0221E+26 * (1.05457266E-34/1.602176634E-19)**2 * (1E+10)**2 * C**2= 1/1.602176634E-19 * C
    C = 1/1.602176634E-19 / (6.0221E+26 * (1.05457266E-34/1.602176634E-19)**2 * (1E+10)**2)
      = 239.22657077980037
    '''

    atom_name = ["$^{14}N$", "$^{13}C(1)$", "$^{13}C(2)$", "$^{13}C(3)$", "$^{13}C(4)$", "$^{13}C(5)$"]

    # 3\times 3\times 3
    # atom_list = [1, 2, 92, 32, 57, 55]

    # 4\times 4\times 4
    # atom_list = [1, 2, 153, 27, 232, 231]

    atom_list = [0,
                 1, 2, 3,
                 152, 155, 279, 293, 408, 420,
                 26, 29, 41,
                 231, 234, 346, 361, 473, 485,
                 230, 345, 469]

    v_parallel = np.array([1, 1, 1])
    v_perpendicular = np.array([1, -1, 0])
    key_list = ["scale", "mode", "displace", "energy", "quadrupole_moment"] + \
               [f"fermi_contact_{atom}" for atom in atom_list] + \
               [f"dipolar_{index}_{atom}" for index in ["xx", "yy", "zz", "xy", "xz", "yz"] for atom in atom_list]

    # -----------------------------------------------------------------------
    # correct data

    # with_error = json.load(open(work_dir + "/fp_{scale}.json", "r"))
    # corrected = json.load(open(work_dir + "/temp.json", "r"))
    # for i in range(len(with_error)):
    #     if with_error[i]["path"]== 'p_29_1.5/OUTCAR':
    #         with_error[i]=corrected[0]
    #         print(with_error[i])
    #         break
    # json.dump(with_error, open(work_dir + "/fp_{scale}.json", "w"))
    # -----------------------------------------------------------
    # read the metadata and pack

    # scale = 0.97
    # value_list = sum([light_outcar(outcar_path, atom_list)
    #                   for outcar_path in [work_dir + f"/static_{scale}.json", work_dir + f"/fp_{scale}.json"]], [])

    # json.dump(value_list, open(work_dir + f"/value_{scale}.json", "w"))
    # ----------------------------------------------------------------------------------------------------------
    # fit
    # value_list = json.load(open(work_dir + f"/value_{scale}.json", "r"))

    # mass, frequency, motion = json.load(open(work_dir + f"/mode_{scale}.json", "r"))
    # fitting = fitting_frozen_phonon_at_fixed_volume(value_list, key_list, scale, frequency, 3, 4)
    # json.dump(fitting, open(work_dir + f"/fitting_{scale}.json", "w"))
    # ---------------------------------------------------------------------------------------------
    # check symmetry
    scale = 1.0
    atom = 0
    value_list = json.load(open(work_dir + f"/value_{scale}.json", "r"))
    dipolar_index = [key_list[3:].index(f"dipolar_{label}_{atom}") for label in ["xx", "yy", "zz", "xy", "xz", "yz"]]
    for i in value_list:
        if i[1] == 0:
            print(i[dipolar_index[0]:dipolar_index[-1] + 1])
    # ---------------------------------------------------------------------------------------------

    # for various volume
    # scale_list = [0.99, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03]
    # frequency_dict=json.load(open(work_dir + "/shifted_frequency_1.0.json", "r"))
    # for scale in scale_list:
    #     print(scale)
    #     value_list = sum([light_outcar(outcar_path, atom_list)
    #                       for outcar_path in [work_dir + f"/static_{scale}.json", work_dir + f"/fp_{scale}.json"]], [])
    #
    #     json.dump(value_list, open(work_dir + f"/value_{scale}.json", "w"))
    # ----------------------------------------------------------------------------------------------------------
    # value_list = json.load(open(work_dir + f"/value_{scale}.json", "r"))
    # frequency = frequency_dict[str(scale)]
    # fitting = fitting_frozen_phonon_at_fixed_volume(value_list, key_list, scale, frequency, 3, 4)
    # json.dump(fitting, open(work_dir + f"/fitting_{scale}.json", "w"))

    # ---------------------------------------------------------------------------------------------
    # display fixed volume
    scale = 1.0

    coef_list, error_list = json.load(open(work_dir + f"/fitting_{scale}.json", "r"))

    # 二阶导和二次项系数差2
    # 在 0 处的二阶导, assuming A = a * x ** 2, A''|0 = 2 * a
    # np.polyval(np.polyder(np.polyder(coef)), 0)
    # for specific displacement, 2 * a = (A[displacement] + A[-displacement] - 2 * A0) / displacement ** 2

    # t[289] == 300
    # t_sample.shape == (490,)
    t_sample = np.linspace(20, 500, 25)
    pressure_sample = np.linspace(-10, 30, 9)

    # coef_list.shape == (1530, 156, 3)
    # second_order_coef.shape == (156, 1530)
    second_order_coef = np.array(coef_list).transpose((2, 1, 0))[0]

    eos = json.load(open(work_dir + "\eos_ptv.json", "r"))

    # ----------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111)
    atom = 26
    fermi_index = key_list[3:].index(f"fermi_contact_{atom}")

    frequency_classic = get_shifted_frequency(1.)
    # 1 / (exp(f / k / T) - 1)
    # number_of_phonon.shape == (490, 1530)
    number_of_phonon_classic = 1 / (np.exp(np.outer(1 / t_sample, frequency_classic[3:]) / K) - 1)
    # A * (n / f) / C
    # Ci = A / f / C
    # (156, 1530) @ (1530, 490) = (156, 490)
    delta_A_classic = second_order_coef @ (number_of_phonon_classic / frequency_classic[3:]).T / C
    ax.plot(t_sample, delta_A_classic[fermi_index], label="classic")

    frequency_zero_point = get_shifted_frequency(find_V(eos, 0, 0))
    number_of_phonon_zero_point = 1 / (np.exp(np.outer(1 / t_sample, frequency_zero_point[3:]) / K) - 1)
    delta_A_zero_point = second_order_coef @ (number_of_phonon_zero_point / frequency_zero_point[3:]).T / C
    ax.plot(t_sample, delta_A_zero_point[fermi_index], label="zero point")

    for p in pressure_sample:
        n_f = []
        for t in t_sample:
            frequency = get_shifted_frequency(find_V(eos, p, t))
            number_of_phonon = 1 / (np.exp(frequency[3:] / (K * t)) - 1)
            n_f.append(number_of_phonon / frequency[3:])
        delta_A = second_order_coef @ np.array(n_f).T / C

        # key_list = ["scale", "mode", "displace", "energy", "quadrupole_moment"] + \
        #            [f"fermi_contact_{atom}" for atom in atom_list] + \
        #            [f"dipolar_{index}_{atom}" for index in ["xx", "yy", "zz", "xy", "xz", "yz"] for atom in atom_list]

        # when number_of_phonon == 0.5 gives zero point

        if round(p) % 10 == 0:
            ax.plot(t_sample, delta_A[fermi_index], label=f"{round(p)} GPa")

    ax.set_xlabel("$T$ (K)")
    ax.set_ylabel("$\delta A$ (MHz)")
    plt.legend()
    plt.show()
    # --------------------------------------------------------------------------------

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # for t in t_sample:
    #     n_f = []
    #     for p in pressure_sample:
    #         frequency = get_shifted_frequency(find_V(eos, p, t))
    #         number_of_phonon = 1 / (np.exp(frequency[3:] / (K * t)) - 1)
    #         n_f.append(number_of_phonon / frequency[3:])
    #     delta_A = second_order_coef @ np.array(n_f).T / C
    #
    #     if round(t) % 100 == 0:
    #         ax.plot(pressure_sample, delta_A[2], label=f"{round(t)} K")
    #
    # ax.set_xlabel("$p$ (GPa)")
    # ax.set_ylabel("$\delta A$ (MHz)")
    # plt.legend()
    # plt.show()
    # -------------------------------------------------------------------

    # A * (n + 1) * n / T / T / K / C
    # (156, 1530) * (490, 1530) * (490, 1530) / (490,) / (490,) / K / C
    # dA_dT = second_order_coef @ ((number_of_phonon * (number_of_phonon + 1)).T / np.power(t_sample, 2)) / K / C
    # ---------------------------------------------------------
    # temperature correction on various volume

    # scale_list = [0.99, 1.0, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03]
    # fermi_contact = {atom: [] for atom in atom_list}
    # dipolar = {atom: [] for atom in atom_list}
    # for scale in scale_list:
    #     # coef_list.shape == (1530, 156, 3)
    #     # key_list = ["energy", "quadrupole_moment"] + \
    #     #            [f"fermi_contact_{atom}" for atom in atom_list] + \
    #     #            [f"dipolar_{index}_{atom}" for index in ["xx", "yy", "zz", "xy", "xz", "yz"] for atom in atom_list]
    #
    #     # for second order
    #     # coef_list, error_list = json.load(open(work_dir + f"/fitting_{scale}.json", "r"))
    #     # (3, 1530, 156)
    #     # coef_list = np.array(coef_list).transpose([2, 0, 1])[0, 0]
    #     # for A
    #     for value in json.load(open(work_dir + f"/value_{scale}.json", "r")):
    #         if value[1:3]==[0,0]:
    #             coef_list = np.array(value[3:])
    #             break
    #
    #     # atom_list = [0,
    #     #              1, 2, 3,
    #     #              152, 155, 279, 293, 408, 420,
    #     #              26, 29, 41,
    #     #              231, 234, 346, 361, 473, 485,
    #     #              230, 345, 469]
    #     for atom in atom_list:
    #         fermi_contact_index = key_list[3:].index(f"fermi_contact_{atom}")
    #         dipolar_index = [key_list[3:].index(f"dipolar_{index}_{atom}")
    #                          for index in ["xx", "yy", "zz", "xy", "xz", "yz"]]
    #         fermi_contact[atom].append(coef_list[fermi_contact_index])
    #         dipolar[atom].append(coef_list[dipolar_index])
    # for atom in dipolar:
    #     dipolar[atom] = np.array(dipolar[atom]).T
    #
    # scale_sample = np.linspace(scale_list[0], scale_list[-1], 100)
    # for atom in atom_list[:1]:
    #     # fermi contact
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.scatter(scale_list, fermi_contact[atom], c="r", marker="x")
    #     coef, error, outer = poly_fitting(scale_list, fermi_contact[atom], 1)
    #     ax.plot(scale_sample, np.poly1d(coef)(scale_sample))
    #     ax.set_xlabel("Scale Factor")
    #     ax.set_ylabel("Hyperfine Coupling (MHz)")
    #     # ax.set_ylabel("Second Order Coefficient")
    #     plt.show()
    #
    #     # dipolar
    #     fig = plt.figure()
    #     for ax_index, xyz_index in zip([321, 322, 323, 324, 325, 326], range(6)):
    #         ax = fig.add_subplot(ax_index)
    #         ax.scatter(scale_list, dipolar[atom][xyz_index], c="r", marker="x")
    #         coef, error, outer = poly_fitting(scale_list, dipolar[atom][xyz_index], 1)
    #         if abs(coef[0] / error[0]) > 8:
    #             ax.plot(scale_sample, np.poly1d(coef)(scale_sample))
    #         else:
    #             ax.plot(scale_list, dipolar[atom][xyz_index])
    #
    #         ymin, ymax = ax.get_ylim()
    #         if abs((ymin + ymax) / (ymin - ymax)) > 40:
    #             offset = round((ymin + ymax) / 2, 2)
    #             ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y', useOffset=offset)
    #         else:
    #             ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
    #         ax.set_title("$A_{" + ["xx", "yy", "zz", "xy", "xz", "yz"][xyz_index] + "}$")
    #     plt.subplots_adjust(hspace=1)
    #     plt.show()
    # -------------------------------------------------------
    # static energy check
    # t_sample = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0]
    # scale_under_t = [1.0036800710943974, 1.0036800710943974, 1.0036800710943974, 1.0037993335701643, 1.0039781741653095,
    #                  1.0042761003146308, 1.0046333785265493, 1.0050498821900886, 1.0055254640657163, 1.0060599565008865,
    #                  1.006593881615961, 1.0072456885137235, 1.0078375090706206]
    # scale_list = []
    # energy_list = []
    # for json_path in os.listdir(work_dir):
    #     if json_path[:6] == "static":
    #         # there are only one point in static.json
    #         static = json.load(open(work_dir + f"/{json_path}", "r"))[0]
    #         scale = float(static["path"].split("/")[0])
    #         energy = static["energy"]
    #         scale_list.append(scale)
    #         energy_list.append(energy)
    # scale_list = np.array(scale_list)
    # argsort = scale_list.argsort()
    # scale_list = scale_list[argsort]
    # arg_standard = np.where(scale_list == 1.0)
    # energy_list = np.array(energy_list)[argsort]
    # energy_list = energy_list - energy_list[arg_standard]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(scale_list, energy_list, marker="x", c="r")
    # ax.plot(scale_list, energy_list)
    # top_ax = ax.twiny()
    #
    # start = 0.94
    # end = 1.06
    #
    # top_ax.set_xticks((np.array([1., scale_under_t[0], scale_under_t[-1]]) - start) / (end - start))
    # top_ax.set_xticklabels(["classic", "$0~K$", "$1200~K$"])
    #
    # plt.setp(top_ax.get_xticklabels(), rotation=60, ha="left")
    # ax.set_xlabel("scale")
    # ax.set_ylabel("Energy (eV)")
    # ax.set_xlim(0.94, 1.06)
    # print(ax.get_xticks(), ax.get_xticklabels())
    # plt.show()
