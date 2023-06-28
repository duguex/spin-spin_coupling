#!/usr/bin/env python
# coding: utf-8

import json
import json_tricks
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import LocalOutlierFactor as LOF

from phonon_frequency import get_shifted_frequency
from thermal_expansion import einstein_model
from read_yaml import construct_matrix, projection
from sympy import symbols, lambdify, Matrix, sign
from uncertainties import unumpy, umath, ufloat
from scc_lib import poly_fitting


def light_outcar(outcar_json_path, atom_list):
    # repack metadata
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


def energy_fitting(displace, energy, mode):
    # fit values with second-order polynominal in value_list
    assert displace.shape[0] == energy.shape[0] > 3, "The size of x and y should be the same and larger than 3 to fit."

    displace = np.array(displace)
    argsort = displace.argsort()
    displace = displace[argsort]
    energy = np.array(energy)[argsort]
    coef, error, outer = poly_fitting(displace, energy, 2)
    second_order_coef = ufloat(coef[0], error[0])
    fitted_frequency = (second_order_coef * 2 / C) ** .5

    # mass check
    # a * x ^ 2 + b * x + c
    # a == frequency ** 2 / 2 * C
    # b == c == 0

    terrible_fit = True if abs(error[0] / coef[0]) > .05 else False
    # theoretical_second_order_coef = frequency ** 2 / 2 * C
    # harmonic_fail = True if abs(coef[0] / theoretical_second_order_coef - 1) > .05 else False

    if terrible_fit:
        plot_mass_fit(displace, energy, coef, fitted_frequency.nominal_value, sample=100)
        print(f"terrible mass fit for mode {mode}: {coef} ± {error}")
        # if harmonic_fail:
        #     print(f"harmonic may fail for mode {mode}: fitted = {coef} ± {error}, "
        #           f"harmonic = {theoretical_second_order_coef}")

    return [fitted_frequency.nominal_value, fitted_frequency.std_dev]


def plot_mass_fit(displace, energy, coef, frequency, sample=100):
    displace_sample = np.linspace(displace[0], displace[-1], sample)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(displace, energy - coef[-1])
    ax.plot(displace_sample, np.poly1d(coef)(displace_sample) - coef[-1], "r--", label="fitting")
    ax.plot(displace_sample, (frequency * displace_sample) ** 2 / 2 * C, label="harmonic")

    # -----------------------------------------------------------------------------------------
    # +声子波函数

    number_of_phonon = 0

    interval = (displace[-1] - displace[0]) / sample
    while True:
        phonon_energy = frequency * (number_of_phonon + 0.5)
        if phonon_energy > min(np.poly1d(coef)([displace[0], displace[-1]])):
            break
        else:
            ax.plot([displace[0], displace[-1]], [phonon_energy, phonon_energy], "--", linewidth=0.7)
            _x = displace_sample * (frequency * C) ** 0.5
            phonon_wf = np.e ** (-_x ** 2 / 2) * \
                        np.polynomial.hermite.hermval(_x, np.append(np.zeros(number_of_phonon), 1))
            # 0.25 跟据显示效果可以调整
            ax.plot(displace_sample, .25 * frequency / (sum(phonon_wf ** 2) * interval) ** 0.5 * phonon_wf +
                    frequency * (number_of_phonon + 0.5), linewidth=0.7)
            number_of_phonon += 1
    # ----------------------------------------------------------------------------------------------

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


def value_fitting(displace_array, value_array):
    # fit values with second-order polynominal in value_list
    # value is already transposed, and the shape is (value, displace)
    assert displace_array.shape[0] == value_array.shape[1] > 3, \
        "The size of x and y should be the same and larger than 3 to fit."

    # fitting displacement and values for specific mode
    coef_list = []
    error_list = []

    for value in value_array:
        coef, error, outer = poly_fitting(displace_array, value, 2)
        coef_list.append(coef.tolist())
        error_list.append(error.tolist())
    return [coef_list, error_list]


def fit_at_fixed_volume(value_list, mode_range):
    """
        某模式单个声子对应的温度 frequency[nmode] / np.log(2) / K K
        振幅 np.sqrt(2 / frequency[nmode] / C) amu^(1/2)A
        300K 对应的声子数 number_of_phonon_for_specfic_mode = 1 / (np.exp(frequency[nmode] / (K * 300)) - 1)
        振幅 np.sqrt(2 * number_of_phonon_for_specfic_mode / frequency[nmode] / C) amu^(1/2)A
    """

    scale_index = key_list.index("scale")
    mode_index = key_list.index("mode")
    displace_index = key_list.index("displace")
    energy_index = key_list.index("energy")
    # in order of "scale", "mode", "displace", "energy"
    # the first three label the configuration, and "energy" is used to do self-consist check

    fp_array = np.array(value_list)
    assert np.all(fp_array.T[scale_index] == fp_array[0][scale_index]), "the scale should be universal in value list."
    mode_array = fp_array.T[mode_index]

    # for each mode there are 1 * energy, 1 * quadrupole_moment for 14N,
    # atom_list * fermi_contact and (atom_list * 6) * dipolar need to fit

    coef_list = []
    error_list = []
    fitted_frequency_list = []
    fitted_frequency_error_list = []

    for mode in mode_range:
        # the shape of value is (displace, values)
        value_at_mode = fp_array[np.where((mode_array == 0) | (mode_array == mode))]
        displace_array = value_at_mode.T[displace_index]
        argsort_displace = displace_array.argsort()

        displace_array = displace_array[argsort_displace]
        value_at_mode = value_at_mode[argsort_displace]

        energy_array = value_at_mode.T[energy_index]
        value_array = value_at_mode.T[value_start_from:]

        fitted_frequency, fitted_frequency_error = energy_fitting(displace_array, energy_array, mode)
        _coef_list, _error_list = value_fitting(displace_array, value_array)

        fitted_frequency_list.append(fitted_frequency)
        fitted_frequency_error_list.append(fitted_frequency_error)
        coef_list.append(_coef_list)
        error_list.append(_error_list)

    return [coef_list, error_list, fitted_frequency_list, fitted_frequency_error_list]


def fitting_static(value_list, key_list):
    scale_index = key_list.index("scale")
    # keys are "scale", "mode", "displace"

    # there are 1 * energy, 1 * quadrupole_moment for 14N,
    # atom_list * fermi_contact and (atom_list * 6) * dipolar need to fit

    _value_list = np.array(value_list).T
    scale_list = _value_list[scale_index] - 1
    coef_list = []
    error_list = []
    for key, value in zip(key_list[value_start_from:], _value_list[value_start_from:]):
        coef, error, outer = poly_fitting(scale_list, value, 1)
        coef_list.append(coef.tolist())
        error_list.append(error.tolist())

    return [coef_list, error_list]


# anharmonic is from anharmonic vibration ax3/<x> and linear x dependent A(x)=ax


def get_scale(eos, p, t):
    for state in eos:
        if state[0] == p and state[1] == t:
            _scale = state[2]
            break
    else:
        print(f"p = {p}, T = {t} not found.")
        _scale = None
    return _scale


def get_index_of_value_list(atom):
    return key_list[value_start_from:].index(f"fermi_contact_{atom}"), \
           [key_list[value_start_from:].index(f"dipolar_{atom}_{index}") for index in
            ["xx", "yy", "zz", "xy", "xz", "yz"]]


def get_A_parameter_from_fitting_sympy(coef_list, atom):
    fermi_index, dipolar_index = get_index_of_value_list(atom)

    scale = symbols("x")
    fermi_contact = np.poly1d(coef_list[fermi_index])(scale)
    dipolar = [np.poly1d(coef_list[index])(scale) for index in dipolar_index]
    A_tensor = Matrix(construct_matrix(fermi_contact, dipolar))
    _direction = Matrix(direction)
    _direction = _direction / _direction.norm()

    A_parameter = (A_tensor * _direction).norm() * sign(fermi_contact)

    return lambdify(scale, A_parameter)


def get_A_parameter_from_fitting_uncertainties(coef_list_with_uncertainties, atom, fermi_only):
    if atom == "quadrupole":
        quadrupole_index = key_list[value_start_from:].index(f"quadrupole_moment")
    else:
        fermi_index, dipolar_index = get_index_of_value_list(atom)

    # the shape of coef_list/error_list is (values, 2)

    def _get_A_parameter(scale):
        if atom == "quadrupole":
            quadrupole = np.poly1d(coef_list_with_uncertainties[quadrupole_index])(scale)
            return quadrupole
        else:
            fermi_contact = np.poly1d(coef_list_with_uncertainties[fermi_index])(scale)
            if fermi_only:
                return fermi_contact
            else:
                dipolar = [np.poly1d(coef_list_with_uncertainties[index])(scale) for index in dipolar_index]
                A_tensor = np.array(construct_matrix(fermi_contact, dipolar))

                _direction = np.array(direction)
                _direction = _direction / np.linalg.norm(_direction)
                # the shape of A_tensor is (3, 3) or (3, 3, scale_sample)
                if A_tensor.shape == (3, 3):
                    # for i in A_tensor:
                    #     for j in i:
                    #         print('{:.1uS}'.format(j), end=" ")
                    #     print()
                    A_parameter = np.sum(A_tensor.dot(_direction) ** 2) ** .5 * np.sign(np.sum(A_tensor))
                else:
                    A_tensor = np.moveaxis(A_tensor, (0, 1), (-2, -1))
                    A_parameter = np.sum(A_tensor.dot(_direction) ** 2, axis=-1) ** .5 * \
                                  np.sign(np.sum(A_tensor, axis=(-2, -1)))
                    # A_parameter = np.linalg.norm(A_tensor.dot(_direction)) * np.sign(A_fermi)

                return A_parameter

    return _get_A_parameter


def get_A_parameter_from_value_list(value_list, atom, fermi_only):
    fermi_index, dipolar_index = get_index_of_value_list(atom)
    fermi_index += value_start_from
    dipolar_index = np.array(dipolar_index) + value_start_from

    fermi_contact = value_list[fermi_index]
    if fermi_only:
        return fermi_contact
    else:
        dipolar = [value_list[index] for index in dipolar_index]
        A_tensor = np.array(construct_matrix(fermi_contact, dipolar))
        _direction = np.array(direction)
        _direction = _direction / np.linalg.norm(_direction)

        # the shape of A_tensor is (3, 3) or (3, 3, scale_sample)
        if A_tensor.shape == (3, 3):
            A_parameter = np.linalg.norm(A_tensor.dot(_direction)) * np.sign(np.sum(A_tensor))
        else:
            A_parameter = np.linalg.norm(A_tensor.transpose([2, 0, 1]).dot(_direction), axis=1) * \
                          np.sign(np.sum(A_tensor, axis=(0, 1)))

        return A_parameter


def get_variation(variation, coef_list, atom, scale, fermi_only):
    # A.shape = (values, T_sample)
    if atom == "quadrupole":
        quadrupole_index = key_list[value_start_from:].index(f"quadrupole_moment")
        quadrupole = variation[quadrupole_index]
        return quadrupole
    else:
        fermi_index, dipolar_index = get_index_of_value_list(atom)
        fermi_contact_variation = variation[fermi_index]
        if fermi_only:
            return fermi_contact_variation
        else:
            dipolar_variation = [variation[index] for index in dipolar_index]
            A_tensor_variation = np.array(construct_matrix(fermi_contact_variation, dipolar_variation))

            if not type(scale) in [list, np.ndarray]:
                scale = np.array(variation.shape[1] * [scale])

            fermi_contact_ref = np.poly1d(coef_list[fermi_index])(scale)
            dipolar_ref = [np.poly1d(coef_list[index])(scale) for index in dipolar_index]
            A_tensor_ref = np.array(construct_matrix(fermi_contact_ref, dipolar_ref))

            A_tensor = A_tensor_ref + A_tensor_variation

            _direction = np.array(direction)
            _direction = _direction / np.linalg.norm(_direction)

            A_parameter = np.linalg.norm(A_tensor.transpose([2, 0, 1]).dot(_direction), axis=1) * \
                          np.sign(np.sum(A_tensor, axis=(0, 1)))

            A_parameter_ref = np.linalg.norm(A_tensor_ref.transpose([2, 0, 1]).dot(_direction), axis=1) * \
                              np.sign(np.sum(A_tensor_ref, axis=(0, 1)))

    return A_parameter - A_parameter_ref


def get_scale_dependent_second_order_coef(coef_list, scale):
    if type(scale) in [np.ndarray, list]:
        scale = np.array(scale)
        # (values, modes, 2) -> (values, 2, modes)
        coef_list = coef_list.transpose(0, 2, 1)
        # (p ,t, values, modes)
        return np.concatenate((scale[..., np.newaxis], np.ones_like(scale)[..., np.newaxis]), axis=-1).dot(coef_list)
    else:
        # (values, modes, 2)
        return coef_list.dot([scale, 1])


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # settings

    # matplotlib.rcParams["backend"] = "PDF"
    # plt.rcParams["font.size"] = 15
    dump_plot_data = False
    fermi_only = False
    fig_type = "png"
    metadata_dir = r"C:\Users\dugue\Desktop\metadata"
    work_dir = r"C:\Users\dugue\OneDrive\spin-spin_coupling\444_520_PS"
    prefix = "611"
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
    # ---------------------------------------------------------------------------------------------
    # atom_list

    atom_name = ["$^{14}N$", "$^{13}C(1)$", "$^{13}C(2)$", "$^{13}C(3)$", "$^{13}C(4)$", "$^{13}C(5)$"]

    # 3\times 3\times 3
    # atom_list = [1, 2, 92, 32, 57, 55]

    # 4\times 4\times 4
    # atom_list = [1, 2, 153, 27, 232, 231]

    # atom_list = [0,
    #              1, 2, 3,
    #              152, 155, 279, 293, 408, 420,
    #              26, 29, 41,
    #              231, 234, 346, 361, 473, 485,
    #              230, 345, 469]

    atom_list = []
    distance_list = []
    atom_dict = json.load(open(work_dir + "/atom_dict.json", "r"))
    for distance, _atom_list in atom_dict.items():
        if float(distance) > 5:
            break
        distance_list.append(float(distance))
        atom_list.append(sorted(_atom_list)[0])
    print("atom_list", atom_list)

    scale_list = [0.99, 1.0, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03]
    direction = [1, 1, 1]
    # direction = [1, -1, 0]

    # -----------------------------------------------------------------------------------------------
    key_list = ["scale", "mode", "displace", "energy", "quadrupole_moment"] + \
               [f"fermi_contact_{atom}" for atom in atom_list] + \
               [f"dipolar_{atom}_{index}" for atom in atom_list for index in ["xx", "yy", "zz", "xy", "xz", "yz"]]
    value_start_from = 3
    number_of_values = len(key_list) - 3
    # -----------------------------------------------------------------------
    # correct data

    # with_error = json.load(open(metadata_dir + f"/fp_{scale}.json", "r"))
    # corrected = json.load(open(work_dir + "/temp.json", "r"))
    # for i in range(len(with_error)):
    #     if with_error[i]["path"]== 'p_29_1.5/OUTCAR':
    #         with_error[i]=corrected[0]
    #         print(with_error[i])
    #         break
    # json.dump(with_error, open(metadata_dir + "/fp_{scale}.json", "w"))
    #
    # corrected = json.load(open(work_dir + "/corr.json", "r"))
    # for scale in [1.03, 0.99, 1.02, 1.005, 1.025, 1.015]:
    #     with_error = json.load(open(metadata_dir + f"/fp_{scale}.json", "r"))
    #     for i in range(len(with_error)):
    #         if with_error[i]["path"] == 'x_29_1.5/OUTCAR':
    #             print(with_error[i]["energy"])
    #             for j in corrected:
    #                 if j["path"] == f"2_6_{scale}/1/x_29_1.5/OUTCAR":
    #                     j["path"] = 'x_29_1.5/OUTCAR'
    #                     print(j["energy"])
    #                     with_error[i] = j
    #                     break
    #             break
    #     json.dump(with_error, open(metadata_dir + f"/fp_{scale}.json", "w"))
    # -----------------------------------------------------------
    # phonon: read the metadata and repack with light_outcar

    # for scale in scale_list:
    #     value_list = sum([light_outcar(outcar_path, atom_list)
    #                       for outcar_path in
    #                       [metadata_dir + f"/static_{scale}.json", metadata_dir + f"/fp_{scale}.json"]], [])
    #
    #     json.dump(value_list, open(work_dir + f"/value_{scale}_{prefix}.json", "w"))
    # -----------------------------------------------------------
    # static: read the metadata and repack with light_outcar

    # static_path_list = []
    # for static_path in os.listdir(metadata_dir):
    #     # if static_path[:7] == "static_":
    #     if static_path[:3] == "a0_" and not "dont_read" in static_path:
    #         static_path_list.append(static_path)
    # print(static_path_list)
    # value_list = sum([light_outcar(metadata_dir + "/" + outcar_path, atom_list) for outcar_path in static_path_list], [])
    #
    # json.dump(value_list, open(work_dir + f"/static_{prefix}_dont_read.json", "w"))
    # ----------------------------------------------------------------------------------------------------------
    # phonon: fit
    # There are two different way to calculate the volume dependence of phonon:
    # 1. calculate phonon independently, and match modes at various volumes (the projection of phonon can be diffusible)
    # 2. calculated normalized atomic motions at a volume, and calculated frequencies at different volume (not the real phonon)

    # for scale in scale_list:
    #     print("scale = ", scale)
    #     value_list = json.load(open(work_dir + f"/value_{scale}_{prefix}.json", "r"))
    #     _, frequency, _ = json.load(open(metadata_dir + f"/mode_{scale}.json", "r"))
    #     coef_list, error_list, fitted_frequency_list, \
    #     fitted_frequency_error_list = fit_at_fixed_volume(value_list, np.arange(3, len(frequency)))
    #     json.dump([coef_list, error_list], open(work_dir + f"/fitting_{prefix}_{scale}.json", "w"))
    #     json.dump([fitted_frequency_list,
    #                fitted_frequency_error_list], open(work_dir + f"/fitted_frequency_{prefix}_{scale}.json", "w"))
    # ----------------------------------------------------------------------------------------------------------
    # phonon: fit volume changes

    # coef_dict = {}
    # for scale in scale_list:
    #     coef_list, error_list = json.load(open(work_dir + f"/fitting_{prefix}_{scale}.json", "r"))
    #     coef_dict[scale] = coef_list
    #
    # scale_array = np.array(scale_list) - 1
    # # coef_array.shape == (scale, mode, value, 3)
    # coef_array = np.array(list(coef_dict.values()))
    # # second_order_coef_array.shape == (mode, value, scale)
    # second_order_coef_array = coef_array.transpose((3, 1, 2, 0))[0]
    # second_order_coef_array = second_order_coef_array.reshape(-1, len(scale_list))
    #
    # coef_list = []
    # error_list = []
    # scale_sample = np.linspace(scale_array[0], scale_array[-1], 100)
    # for second_order_coef_vs_scale in second_order_coef_array:
    #     coef, error, outer = poly_fitting(scale_array, second_order_coef_vs_scale, 1)
    #     coef_list.append(coef.tolist())
    #     error_list.append(error.tolist())
    #
    #     # plt.scatter(scale_array, second_order_coef_vs_scale)
    #     # plt.plot(scale_sample, np.poly1d(coef)(scale_sample))
    #     # plt.show()
    # json.dump([coef_list, error_list], open(work_dir + f"/fitting_second_order_vs_scale_{prefix}.json", "w"))
    # ----------------------------------------------------------------------------------------------------------
    # static: fit
    # new but bad
    # value_list = json.load(open(work_dir + f"/static.json", "r"))
    # fitting = fitting_static(value_list, key_list)
    # json.dump(fitting, open(work_dir + f"/fitting_static.json", "w"))

    # old but good
    # value_list = json.load(open(work_dir + f"/static_{prefix}_dont_read.json", "r"))
    # fitting = fitting_static(value_list, key_list)
    # json.dump(fitting, open(work_dir + f"/fitting_static_{prefix}.json", "w"))
    # ----------------------------------------------------------------------------------------------------------
    # static: linear coefficient and zero-point correction

    # pv_0K = json.load(open(work_dir + f"\pv_0K_{prefix}.json", "r"))
    # _pressure_sample = np.array(pv_0K["pressure"])
    # _p_index_0 = np.where(_pressure_sample == 0)[0][0]
    # _classical = np.array(pv_0K["classical"]) - 1
    # _phonon = np.array(pv_0K["phonon"]) - 1
    #
    # # another one: work_dir + f"/fitting_static.json"
    # static_coef_list, static_error_list = json.load(open(work_dir + f"/fitting_static_{prefix}.json", "r"))
    # static_coef_list_with_uncertainties = unumpy.uarray(static_coef_list, static_error_list)
    #
    # get_A_parameter_uncertainties = get_A_parameter_from_fitting_uncertainties(
    #     static_coef_list_with_uncertainties, 0, fermi_only=False)
    #
    # deltaA_0_attribute_to_zp_volume = get_A_parameter_uncertainties(_phonon) - get_A_parameter_uncertainties(_classical)
    # -------------------------------------------------------------------------------------------------------------------
    # A_cls with uncertainties

    # another choices
    # value_list = json.load(open(work_dir + f"/static.json", "r"))
    # coef_list, error_list = json.load(open(work_dir + f"/fitting_static.json", "r"))

    # value_list = json.load(open(work_dir + f"/static_{prefix}_dont_read.json", "r"))
    # scale_index = key_list.index("scale")
    # _value_list = np.array(value_list).T
    # _scale_list = _value_list[scale_index] - 1
    # static_coef_list, static_error_list = json.load(open(work_dir + f"/fitting_static_{prefix}.json", "r"))
    # static_coef_list_with_uncertainties = unumpy.uarray(static_coef_list, static_error_list)
    #
    # # plot A-A_cls versus delta L/L_cls
    # for atom in atom_list:
    #     get_A_parameter_uncertainties = get_A_parameter_from_fitting_uncertainties(static_coef_list_with_uncertainties,
    #                                                                                atom, fermi_only=fermi_only)
    #     A_cls = get_A_parameter_uncertainties(0).nominal_value
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.scatter(_scale_list * 100, get_A_parameter_from_value_list(_value_list, atom, fermi_only=fermi_only) - A_cls,
    #                marker="x", color="r")
    #     scale_sample = np.linspace(_scale_list[0], _scale_list[-1], 100)
    #     A_parameter_uncertainties = get_A_parameter_uncertainties(scale_sample)
    #     scale_sample_100 = scale_sample * 100
    #     ax.plot(scale_sample_100, unumpy.nominal_values(A_parameter_uncertainties) - A_cls)
    #     ax.plot(scale_sample_100, unumpy.nominal_values(A_parameter_uncertainties) - A_cls
    #             + unumpy.std_devs(A_parameter_uncertainties), "b--")
    #     ax.plot(scale_sample_100, unumpy.nominal_values(A_parameter_uncertainties) - A_cls
    #             - unumpy.std_devs(A_parameter_uncertainties), "b--")
    #     ax.set_xlabel(r'$\frac{\delta L}{L_{\rm cls}}$ (%)', fontsize=15)
    #     ax.set_ylabel(r"$A_0(V)-A_0(V_{\rm cls})$ (MHz)", fontsize=15)
    #
    #     ax.minorticks_on()
    #     ax.xaxis.set_major_locator(MaxNLocator(5))
    #     ax.yaxis.set_major_locator(MaxNLocator(5))
    #     ax.tick_params(axis='both', which='major', labelsize=15)
    #     fig.tight_layout()
    #     plt.savefig(work_dir + f"/lattice_fit_fermi={fermi_only}_{atom}.png", dpi=600)
    #     plt.close("all")
    # ---------------------------------------------------------------------------------------------
    # get scale from (T, P) sample

    pressure_sample = np.linspace(-10, 30, 81)
    t_sample = np.linspace(10, 500, 50)
    eos = json.load(open(work_dir + f"\eos_ptv_81_121_{prefix}.json", "r"))
    p_index_0 = np.where(pressure_sample == 0)[0][0]

    scale_cls = 0
    scale_zp = get_scale(eos, 0, 0) - 1
    # scale_p_t.shape == (pressure_sample, t_sample)
    scale_p_t = [[get_scale(eos, p, t) - 1 for t in t_sample] for p in pressure_sample]

    frequency_coef_list, _ = json.load(
        open(work_dir + f"/shifted_frequency_fitting_with_const_motion_{prefix}.json", "r"))
    # coef_with_uncertainties = unumpy.uarray(coef_list, error_list)
    # (2, phonon)
    frequency_coef_list = np.array(frequency_coef_list).T

    frequency_cls = get_shifted_frequency(frequency_coef_list, scale_cls)
    number_of_modes = frequency_cls.shape[0]

    if dump_plot_data:
        # ----------------------------------------------------------------------------------
        # display phonon contribution
        # 二阶导和二次项系数差2
        # 在 0 处的二阶导, assuming A = a * x ** 2, A''|0 = 2 * a
        # np.polyval(np.polyder(np.polyder(coef)), 0)
        # for specific displacement, 2 * a = (A[displacement] + A[-displacement] - 2 * A0) / displacement ** 2

        # get second_order_coef from fixed volume
        # coef_list.shape == (modes, values, 3)
        # second_order_coef.shape == (values, modes)

        # coef_list, error_list = json.load(open(work_dir + f"/fitting_{scale}.json", "r"))
        # second_order_coef = np.array(coef_list).transpose((2, 1, 0))[0]

        # get_shifted_frequency(work_dir + "/shifted_frequency_fitting.json", 0)[3:]
        frequency_zp = get_shifted_frequency(frequency_coef_list, scale_zp)
        # frequency_p_t.shape == (pressure_sample, t_sample, phonon)
        frequency_p_t = get_shifted_frequency(frequency_coef_list, scale_p_t)
        # frequency fixed
        # frequency_p_t = np.array([[frequency_zp for t in t_sample] for p in pressure_sample])

        # 1 / (exp(f / k / T) - 1)
        # number_of_phonon.shape == (temperature, modes)
        number_of_phonon_cls = 1 / (np.exp(np.outer(1 / t_sample, frequency_cls) / K) - 1)
        number_of_phonon_zp = 1 / (np.exp(np.outer(1 / t_sample, frequency_zp) / K) - 1)
        # (pressure_sample, t_sample, modes)
        number_of_phonon_p_t = (1 / (np.exp(frequency_p_t.transpose(0, 2, 1) / (K * t_sample)) - 1)).transpose(0, 2, 1)

        # number_of_phonon = 0.5
        number_of_phonon_cls_zero = np.ones_like(number_of_phonon_cls) * .5
        number_of_phonon_zp_zero = np.ones_like(number_of_phonon_zp) * .5
        number_of_phonon_p_t_zero = np.ones_like(number_of_phonon_p_t) * .5

        # (values, modes) includes 14N quadrupole

        second_order_coef_list, _ = json.load(open(work_dir + f"/fitting_second_order_vs_scale_{prefix}.json", "r"))
        second_order_coef_list = np.array(second_order_coef_list).reshape(number_of_modes, number_of_values,
                                                                          2).transpose(1, 0, 2)
        # second_order_coef.shape == (modes, values)

        second_order_coef_cls = get_scale_dependent_second_order_coef(second_order_coef_list, scale_cls)
        second_order_coef_zp = get_scale_dependent_second_order_coef(second_order_coef_list, scale_zp)
        # (pressure_sample, t_sample, 2) @ (values, 2, modes) == (pressure_sample, t_sample, values, modes)
        # when number_of_phonon == 0.5 gives zero point
        second_order_coef_p_t = get_scale_dependent_second_order_coef(second_order_coef_list, scale_p_t)

        # A * (n / f) / C
        # Ci = A / f / C
        # (values, modes) @ (modes, temperature) = (values, temperature)
        delta_A_cls = second_order_coef_cls @ (number_of_phonon_cls / frequency_cls).T / C
        delta_A_zp = second_order_coef_zp @ (number_of_phonon_zp / frequency_zp).T / C

        delta_A_cls_zero = second_order_coef_cls @ (number_of_phonon_cls_zero / frequency_cls).T / C
        delta_A_zp_zero = second_order_coef_zp @ (number_of_phonon_zp_zero / frequency_zp).T / C
        # ------------------------------------------------------------------------
        # ckeck product

        # (pressure_sample, t_sample, values, modes) @ (pressure_sample, t_sample, phonon) = (pressure_sample, t_sample, values)
        # delta_A_p_t1 = np.array(
        #     [[second_order_coef_p_t[p_index][t_index] @ (number_of_phonon_p_t / frequency_p_t)[p_index][t_index]
        #       for t_index in range(len(t_sample))] for p_index in range(len(pressure_sample))]) / C
        # delta_A_p_t2 = np.einsum("ijkl,ijl->ijk", second_order_coef_p_t, (number_of_phonon_p_t / frequency_p_t)) / C
        # print(np.allclose(delta_A_p_t1, delta_A_p_t2))
        # --------------------------------------------------------------------------------------------------
        # (pressure_sample, t_sample, values, modes) @ (pressure_sample, t_sample, phonon) = (pressure_sample, t_sample, values)
        delta_A_p_t = np.einsum("ijkl,ijl->ijk", second_order_coef_p_t, (number_of_phonon_p_t / frequency_p_t)) / C
        delta_A_p_t_zero = np.einsum("ijkl,ijl->ijk", second_order_coef_p_t,
                                     (number_of_phonon_p_t_zero / frequency_p_t)) / C

        # use second_order_coef_zp (values, modes) @ (pressure_sample, t_sample, phonon) = (pressure_sample, t_sample, values)
        # delta_A_p_t = np.einsum("ij,klj->kli", second_order_coef_zp, (number_of_phonon_p_t / frequency_p_t)) / C
        # ----------------------------------------------------------------------------------
        # plot A_ph vs temperature

        # the scale dependent A0(V)
        static_coef_list, static_error_list = json.load(open(work_dir + f"/fitting_static_{prefix}.json", "r"))
        static_coef_list_with_uncertainties = unumpy.uarray(static_coef_list, static_error_list)

        plot_data = {}

        # hyperfine
        if fermi_only:
            atom_list = atom_list + ["quadrupole"]
        for atom in atom_list:
            plot_data[atom] = {}

            # A_0
            get_A_parameter_uncertainties = get_A_parameter_from_fitting_uncertainties(
                static_coef_list_with_uncertainties, atom, fermi_only=fermi_only)

            A_0_cls = get_A_parameter_uncertainties(scale_cls).nominal_value
            A_0_ref = A_0_cls
            A_0_zp = np.ones_like(t_sample) * get_A_parameter_uncertainties(scale_zp).nominal_value
            A_0_p_t = unumpy.nominal_values(get_A_parameter_uncertainties(scale_p_t))

            A_0_zp = A_0_zp - A_0_ref
            A_0_p_t = A_0_p_t - A_0_ref
            A_0_cls = np.ones_like(t_sample) * 0

            # A_ph
            A_ph_cls = get_variation(delta_A_cls, None if fermi_only else static_coef_list, atom, scale_cls,
                                     fermi_only=fermi_only)
            A_ph_zp = get_variation(delta_A_zp, None if fermi_only else static_coef_list, atom, scale_zp,
                                    fermi_only=fermi_only)

            A_zero_cls = get_variation(delta_A_cls_zero, None if fermi_only else static_coef_list, atom, scale_cls,
                                       fermi_only=fermi_only)
            A_zero_zp = get_variation(delta_A_zp_zero, None if fermi_only else static_coef_list, atom, scale_zp,
                                      fermi_only=fermi_only)

            # A_0(V_cls) as A_ref
            A_cls = (A_0_cls + A_0_p_t[p_index_0] - A_0_p_t[p_index_0][0]) + A_ph_cls + A_zero_cls
            A_zp = (A_0_zp + A_0_p_t[p_index_0] - A_0_p_t[p_index_0][0]) + A_ph_zp + A_zero_zp

            # A_ph_p_t_mesh.shape == (p, t)
            A_ph_p_t_mesh = []
            A_zero_p_t_mesh = []
            for p_index in range(len(pressure_sample)):
                A_ph_p_t = get_variation(delta_A_p_t[p_index].T, None if fermi_only else static_coef_list, atom,
                                         scale_p_t[p_index],
                                         fermi_only=fermi_only)
                A_zero_p_t = get_variation(delta_A_p_t_zero[p_index].T, None if fermi_only else static_coef_list, atom,
                                           scale_p_t[p_index], fermi_only=fermi_only)

                A_ph_p_t_mesh.append(A_ph_p_t)
                A_zero_p_t_mesh.append(A_zero_p_t)

            A_p_t = A_0_p_t + A_ph_p_t_mesh + A_zero_p_t_mesh

            plot_data[atom] = {"Aref": A_0_ref, "A0_cls": A_0_cls, "A0_zp": A_0_zp, "A0_p_t": A_0_p_t,
                               "Aph_cls": A_ph_cls, "Azero_cls": A_zero_cls, "Aph_zp": A_ph_zp, "Azero_zp": A_zero_zp,
                               "Aph_p_t": A_ph_p_t_mesh, "Azero_p_t": A_zero_p_t_mesh, "A_cls": A_cls, "A_zp": A_zp,
                               "A_p_t": A_p_t}

        json_tricks.dump(plot_data, open(work_dir + f"/plot_test_fermi_data_{prefix}.json", "w"))


    # ----------------------------------------------------------------------------------------

    def get_y_by_tag(x, y, tag=None):
        if tag == "delta":
            return y - y[0]
        elif tag == "deltap":
            return y - y[p_index_0]
        elif tag == "dt":
            print(np.gradient(y, x, edge_order=2) * 1e6)
            return np.gradient(y, x, edge_order=2) * 1e6
        elif tag == "dp":
            print(np.gradient(y, x, edge_order=2))
            return np.gradient(y, x, edge_order=2)
        elif tag == "dt2":
            dy_dx = np.gradient(y, x, edge_order=2)
            return np.gradient(dy_dx, x, edge_order=2) * 1e6
        else:
            return y


    # plot_data = json_tricks.load(open(work_dir + f"/plot_data_{prefix}.json", "r"))
    plot_data = json_tricks.load(open(work_dir + (f"/plot_test_fermi_data_{prefix}.json" if fermi_only
                                                  else f"/plot_test_data_{prefix}.json"), "r"))
    # plot_data = json_tricks.load(open(work_dir + f"/plot_vertical_data_{prefix}.json", "r"))

    # y_label_and_fig_name = {('A0', 'dt'): [r'$\frac{\mathrm{d}A_0}{\mathrm{d}T}$ (Hz/K)', '/dA_0_dT'],
    #                         ('A0', 'delta'): [r'$A_0-A_0(0, P)$ (MHz)', '/deltaA_0'],
    #                         ('A0', 'still'): [r'$A_0-A_0(V_{\rm cls})$ (MHz)', '/A_0'],
    #
    #                         ('Aph', 'dt'): [r'$\frac{\mathrm{d}A_{\rm ph}}{\mathrm{d}T}$ (Hz/K)', '/dA_ph_dT'],
    #                         ('Aph', 'still'): [r'$A_{\rm ph}$ (MHz)', '/A_ph'],
    #
    #                         ('Azero', 'dt'): [r'$\frac{\mathrm{d}A_{\rm ph, zp}}{\mathrm{d}T}$ (Hz/K)',
    #                                           '/dA_ph_zp_dT'],
    #                         ('Azero', 'delta'): [r'$\delta A_{\rm ph, zp}$ (MHz)', '/deltaA_ph_zp'],
    #                         ('Azero', 'still'): [r'$A_{\rm ph, zp}$ (MHz)', '/A_ph_zp'],
    #
    #                         ('A_', 'dt'): [r'$\frac{\mathrm{d}A}{\mathrm{d}T}$ (Hz/K)', '/dA_dT'],
    #                         ('A_', 'delta'): [r'$A-A(0, P)$ (MHz)', '/deltaA'],
    #                         ('A_', 'still'): [r'$A-A_0(V_{\rm cls})$ (MHz)', '/A'],
    #
    #                         ('A_', 'dt2'): [r'$\frac{\mathrm{d}^2A}{\mathrm{d}T^2}$ (Hz/$\rm {K}^2$)', '/d2A_dT2'],
    #                         ('A_', 'dtdp'): [r'$\frac{\partial^2A}{\partial T\partial P}$ (Hz/K/GPa)', '/d2A_dTdP'],
    #
    #                         ('A0', 'dp'): [r'$\frac{\mathrm{d}A_0}{\mathrm{d} P}$ (MHz/GPa)', '/dA_0_dP'],
    #                         ('Aph', 'dp'): [r'$\frac{\mathrm{d}A_{\rm ph}}{\mathrm{d} P}$ (MHz/GPa)', '/dA_ph_dP'],
    #                         ('Azero', 'dp'): [r'$\frac{\mathrm{d}A_{\rm ph, zp}}{\mathrm{d} P}$ (MHz/GPa)',
    #                                           '/dA_ph_zp_dP'],
    #                         ('A_', 'dp'): [r'$\frac{\mathrm{d}A}{\mathrm{d} P}$ (MHz/GPa)', '/dA_dP'],
    #
    #                         ('A0', 'p'): [r'$A_0-A_0(V_{\rm cls})$ (MHz)', '/A_0_P'],
    #                         ('Aph', 'p'): [r'$A_{\rm ph}$ (MHz)', '/A_ph_P'],
    #                         ('Azero', 'p'): [r'$A_{\rm ph, zp}$ (MHz)', '/A_ph_zp_P'],
    #                         ('A_', 'p'): [r'$A-A_0(V_{\rm cls})$ (MHz)', '/A_P'],
    #
    #                         ('A0', 'deltap'): [r'$A_0-A_0(T, 0)$ (MHz)', '/deltaA_0_P'],
    #                         ('Azero', 'deltap'): [r'$\delta A_{\rm ph, zp}$ (MHz)', '/deltaA_ph_zp_P'],
    #                         ('A_', 'deltap'): [r'$A-A_0(T, 0)$ (MHz)', '/deltaA_P'], }
    y_label_and_fig_name = {('A_', 'dtdp'): [r'$\frac{\partial^2A}{\partial T\partial P}$ (Hz/K/GPa)', '/d2A_dTdP']}


    # vertical direction (1, -1, 0)
    # y_label_and_fig_name = {('A_', 'dt2'): [r'$\frac{\mathrm{d}^2A}{\mathrm{d}T^2}$ (Hz/$\rm {K}^2$)', '/d2Ac_dT2'],
    #                         ('A_', 'dt'): [r'$\frac{\mathrm{d}A}{\mathrm{d}T}$ (Hz/K)', '/dAc_dT'], }

    def plot_A(part, tag):
        for atom in plot_data:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            if tag == "dtdp":
                ax.set_xlabel("$T$ (K)")
                A_p_t_mesh = plot_data[atom]["A_p_t"]
                dA_dp = np.gradient(A_p_t_mesh, pressure_sample, edge_order=2, axis=0)

                # exchange check for d2A_dpdt
                # dA_dt = np.gradient(A_p_t_mesh, t_sample, edge_order=2, axis=1) * 1e6
                # d2A_dtdp = np.gradient(dA_dt, pressure_sample, edge_order=2, axis=0)
                # print(np.allclose(d2A_dpdt, d2A_dtdp))

                d2A_dpdt = np.gradient(dA_dp, t_sample, edge_order=2, axis=1) * 1e6

                # plot
                for p_index in range(len(pressure_sample)):
                    if pressure_sample[p_index] % 10 == 0.:
                        label = f"{round(pressure_sample[p_index])} GPa"
                        ax.plot(t_sample, d2A_dpdt[p_index], label=label)

            elif tag in ["dp", "p", "deltap"]:
                ax.set_xlabel("$P$ (GPa)")
                for obj_name in plot_data[atom]:
                    if part in obj_name and "p_t" in obj_name:
                        for t_index in range(len(t_sample)):
                            if t_sample[t_index] % 100 == 0. or t_sample[t_index] == 10:
                                label = f"{round(t_sample[t_index])} K"
                                ax.plot(pressure_sample,
                                        get_y_by_tag(pressure_sample, np.array(plot_data[atom][obj_name]).T[t_index],
                                                     tag=tag), label=label)
            else:
                ax.set_xlabel("$T$ (K)")
                for obj_name in plot_data[atom]:
                    if part in obj_name:
                        if "p_t" in obj_name:
                            for p_index in range(len(pressure_sample)):
                                if pressure_sample[p_index] % 10 == 0.:
                                    label = f"{round(pressure_sample[p_index])} GPa"
                                    ax.plot(t_sample,
                                            get_y_by_tag(t_sample, plot_data[atom][obj_name][p_index], tag=tag),
                                            label=label)
                        else:
                            # obj_name instance: A_0_cls
                            label = obj_name.replace(part if part[-1] == "_" else part + "_", "")
                            ax.plot(t_sample, get_y_by_tag(t_sample, plot_data[atom][obj_name], tag=tag), label=label,
                                    linestyle="--")

            ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')
            plt.legend()

            y_label, fig_name = y_label_and_fig_name[(part, tag)]
            fig_name = work_dir + fig_name + f"_{atom}.png"

            ax.set_ylabel(y_label)
            # fig.tight_layout()
            plt.savefig(fig_name, dpi=600)
            plt.close("all")


    # for part, tag in y_label_and_fig_name:
    #     plot_A(part, tag)

    # heatmap

    import matplotlib.colors as colors

    for atom in atom_list:
        A_p_t_mesh = plot_data[str(atom)]["A_p_t"]
        dA_dp = np.gradient(A_p_t_mesh, pressure_sample, edge_order=2, axis=0)
        dA_dt = np.gradient(A_p_t_mesh, t_sample, edge_order=2, axis=1) * 1e6
        d2A_dpdt = np.gradient(dA_dp, t_sample, edge_order=2, axis=1) * 1e6

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # mesh_plot = np.abs(d2A_dpdt / dA_dt)
        # norm = colors.LogNorm(vmin=mesh_plot.min(), vmax=mesh_plot.max())
        # im = ax.imshow(mesh_plot, norm=norm)
        # ax.set_xlabel("$T$ (K)")
        # ax.set_ylabel("$P$ (GPa)")
        # ax.set_xticks(np.arange(0, 50, 5), np.arange(0, 451, 50))
        # ax.set_yticks(np.arange(0, 90, 10), np.arange(-10, 31, 5))
        # cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(r'$\frac{\partial^2A}{\partial T\partial P}/\frac{\partial A}{\partial T} ({\rm GPa}^{-1})$')
        # ax.set_aspect('auto')
        # fig_name = work_dir + f"/d2A_dTdP_norm_heatmap_{atom}.png"
        # plt.savefig(fig_name, dpi=600)
        # plt.close("all")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        mesh_plot = np.abs(d2A_dpdt / dA_dp * 1e-6)
        norm = colors.LogNorm(vmin=mesh_plot.min(), vmax=mesh_plot.max())
        im = ax.imshow(mesh_plot, norm=norm)
        ax.set_xlabel("$T$ (K)")
        ax.set_ylabel("$P$ (GPa)")
        ax.set_xticks(np.arange(0, 50, 5), np.arange(0, 451, 50))
        ax.set_yticks(np.arange(0, 90, 10), np.arange(-10, 31, 5))
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r'$\frac{\partial^2A}{\partial T\partial P}/\frac{\partial A}{\partial P} ({\rm K}^{-1})$')
        ax.set_aspect('auto')
        fig_name = work_dir + f"/d2A_dTdP_norm_P_heatmap_{atom}.png"
        plt.savefig(fig_name, dpi=600)
        plt.close("all")
    # --------------------------------------------------------------------------------

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # for t in t_sample:
    #     n_f = []
    #     for p in pressure_sample:
    #         frequency = get_shifted_frequency(get_scale(eos, p, t))
    #         number_of_phonon = 1 / (np.exp(frequency[3:] / (K * t)) - 1)
    #         n_f.append(number_of_phonon / frequency[3:])
    #     delta_A = second_order_coef @ np.array(n_f).T / C
    #
    #     if round(t) % 100 == 0:
    #         ax.plot(pressure_sample, delta_A[fermi_contact_index] + delta_A[dipolar_zz_index], label=f"{round(t)} K")
    #
    # ax.set_xlabel("$p$ (GPa)")
    # ax.set_ylabel("$\delta A$ (MHz)")
    # plt.legend()
    # plt.show()
    # -------------------------------------------------------------------
    # A * (n + 1) * n / T / T / K / C
    # (values, modes) * (temperature, modes) * (temperature, modes) / (temperature,) / (temperature,) / K / C
    # dA_dT = second_order_coef @ ((number_of_phonon * (number_of_phonon + 1)).T / np.power(t_sample, 2)) / K / C
    # ---------------------------------------------------------
    # temperature correction on various volume

    # scale_list = [0.99, 1.0, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03]
    # fermi_contact = {atom: [] for atom in atom_list}
    # dipolar = {atom: [] for atom in atom_list}
    # for scale in scale_list:
    #     # for second-order term
    #     # coef_list.shape == (modes, values, 3) -> (3, modes, values)
    #     # coef_list, error_list = json.load(open(work_dir + f"/fitting_{scale}.json", "r"))
    #     # coef_list = np.array(coef_list).transpose([2, 0, 1])[0, 0]
    #     # for A
    #     for value in json.load(open(work_dir + f"/value_{scale}.json", "r")):
    #         if value[1:3] == [0, 0]:
    #             coef_list = np.array(value[3:])
    #             break
    #     for atom in atom_list:
    #         fermi_contact_index, dipolar_index = get_index_of_value_list(atom)
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
    # for json_path in os.listdir(metadata_dir):
    #     if json_path[:6] == "static":
    #         # there are only one point in static.json
    #         static = json.load(open(metadata_dir + f"/{json_path}", "r"))[0]
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
