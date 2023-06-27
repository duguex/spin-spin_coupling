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
from sympy import *
from sympy.plotting import plot
import json
from scipy import stats
import pprint
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator
import json_tricks
from scipy.optimize import curve_fit
from read_yaml import read_dos


def einstein_model(mode):
    t, x, theta = symbols("T X theta")
    # 0-1000K
    c1 = np.array([0.0096, 0.2656, 2.6799, 2.3303]) * 1e-6
    c2 = np.array([159.3, 548.5, 1237.9, 2117.8])
    Einstein_function = x * (theta / t) ** 2 * E ** (theta / t) / (E ** (theta / t) - 1) ** 2
    integrated_Einstein_function = integrate(Einstein_function, (t))
    derivative_Einstein_function = diff(Einstein_function, t, 1)
    # einstein_model(1) alpha 0-5 is not good
    # einstein_model(2) delta L/L(T) 0-2 is not good
    return {0: Einstein_function,
            1: lambdify(t, sum([Einstein_function.subs(x, c1[n]).subs(theta, c2[n]) for n in range(4)]), "numpy"),
            2: lambdify(t, sum([integrated_Einstein_function.subs(x, c1[n]).subs(theta, c2[n]) for n in range(4)]),
                        "numpy"),
            3: lambdify(t, sum([derivative_Einstein_function.subs(x, c1[n]).subs(theta, c2[n]) for n in range(4)]),
                        "numpy")}[mode]


if False:
    # 1-1200
    calculated_thermal = 1e-3 * np.array(json.load(open("thermal_expansion.json", "r"))["da_a"][5:500])
    T = np.linspace(6, 500, 495)
    coef, _ = curve_fit(lambda x, a, b, c: a * x ** 4 +
                                           b * x ** 3 +
                                           c * x ** 2,
                        T, calculated_thermal)
    coef = coef.tolist()
    coef.append(0)
    coef.append(0)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot1 = ax1.plot(T, 1e6 * einstein_model(1)(T), label=r"$\alpha$", color="r", linestyle="dashed")
    e1 = np.poly1d(np.polyder(coef))
    plot4 = ax1.plot(T, 1e6 * e1(T), label=r"$\alpha$", color="r")

    ax1.set_xlabel("$T$ (K)", fontsize=15)
    ax1.set_ylabel(r"$\alpha~(\rm ×10^{-6}~K^{-1})$", fontsize=15)
    ax2 = ax1.twinx()
    plot2 = ax2.plot(T, 1e3 * einstein_model(2)(T), label=r"$\frac{\delta a}{a}$", color="b", linestyle="dashed")
    e2 = np.poly1d(coef)
    plot3 = ax2.plot(T, 1e3 * e2(T), label=r"$calculated \frac{\delta a}{a}$", color="b")
    ax2.scatter(T, 1e3 * calculated_thermal, c="b", s=0.5)
    ax2.set_ylabel(r"$\frac{\delta a}{a}$ (×10$^{-3}$)", fontsize=15)
    plots = plot1 + plot2
    labels = [aplot.get_label() for aplot in plot1 + plot2]
    ax1.legend(plots, labels, fontsize=15)
    ax1.minorticks_on()
    ax2.minorticks_on()
    # ax2.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(work_dir + "/expand_temp." + fig_type, dpi=600, format=fig_type)
    plt.close("all")


def Gauss(x, xc, A, sigma):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - xc) ** 2 / (2 * sigma ** 2))


def fitting(xarray, yarray, fitting_order, check=False, plot=False):
    xarray = xarray if type(xarray) == list else xarray.tolist()
    yarray = yarray if type(yarray) == list else yarray.tolist()
    if not check:
        coef, cov = np.polyfit(xarray, yarray, fitting_order, cov=True)
    else:
        xarray = xarray.copy()
        yarray = yarray.copy()
        xdel = []
        ydel = []
        while True:
            coef, cov = np.polyfit(xarray, yarray, fitting_order, cov=True)
            ynorm = yarray - np.poly1d(coef)(xarray)
            mean, std = ynorm.mean(), ynorm.std()
            test = stats.kstest(ynorm, "norm", (mean, std))
            # print(test)
            for index in range(len(ynorm)):
                if np.abs(ynorm[index] - mean) > 3 * std:
                    xdel.append(xarray.pop(index))
                    ydel.append(yarray.pop(index))
                    break
            else:
                break

    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.scatter(xarray, yarray)
        x = np.linspace(xarray[0], xarray[-1], 100)
        ax1.plot(x, np.poly1d(coef)(x))
        ax2 = fig.add_subplot(122)
        ax2.scatter(xarray, ynorm)
        ax2.plot([x[0], x[-1]], [mean - 3 * std, mean - 3 * std], "b--")
        ax2.plot([x[0], x[-1]], [mean, mean], "k")
        ax2.plot([x[0], x[-1]], [mean + 3 * std, mean + 3 * std], "b--")
        if check and xdel:
            xdel = np.array(xdel)
            ydel = np.array(ydel)
            ax1.scatter(xdel, ydel, marker="x", color="r")
            ax2.scatter(xdel, ydel - np.poly1d(coef)(xdel), marker="x", color="r")

        ax1.set_xlabel(r'$X$ ($\rm \sqrt{amu}\cdot\AA$)', fontsize=15)
        ax2.set_xlabel(r'$X$ ($\rm \sqrt{amu}\cdot\AA$)', fontsize=15)
        ax1.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
        ax2.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
        ax1.set_ylabel("$A$ (MHz)", fontsize=15)
        ax2.set_ylabel("$A-A_{fit}$ (MHz)", fontsize=15)
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax1.yaxis.set_major_locator(MaxNLocator(5))
        plt.xlabel(r'$X$ ($\rm \sqrt{amu}\cdot\AA$)', fontsize=15)
        plt.tight_layout()
        plt.savefig(work_dir + f"/{plot}.png", dpi=600, format="png")
        plt.close("all")

    return {"fit": list(coef),
            "error": [np.sqrt(cov[i][i]) for i in range(len(coef))]}


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

    # matplotlib.rcParams["backend"] = "PDF"
    # plt.rcParams["font.size"] = 15
    dump_fig = True
    dump_json = False
    fig_type = "png"
    # work_dir = r"C:\Users\dugue\OneDrive\spin-spin\444"
    # work_dir = r"C:\Users\dugue\OneDrive\spin-spin\333"
    work_dir = "C:/Users/dugue/OneDrive/spin-spin/1226_strain"
    K = 8.6173e-05
    C = 239.203931
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

    # 4\times 4\times 4
    # atom_list = [1, 2, 153, 27, 232, 231]
    # atom_list = [27]

    # atom_list = [1]

    atom_list = [1, 3, 421, 42, 362, 231]

    # atom_list = [153]

    # symmetry test
    # atom_list = [1,
    #              2, 3, 4,
    #              153, 156, 280, 294, 409, 421,
    #              27, 30, 42,
    #              232, 235, 347, 362, 474, 486,
    #              231, 346, 470]

    # expand flag
    including_expand = True
    including_vib = False
    # eig = -1
    # col = 2
    # row = 2

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

        # expand for 400 PE
        # expand_520_ps for 520 PS
        A_expand_data.sort(key=lambda obj: obj["length_of_vectors"])

        length_of_vectors = [obj['length_of_vectors'] for obj in A_expand_data]
        # length_of_vectors_0 = length_of_vectors[0]
        index_0 = len(length_of_vectors) // 2
        length_of_vectors = np.array(length_of_vectors) / length_of_vectors[index_0] - 1
        # energy = np.array([obj['energy'] for obj in A_expand_data])
        A_expand_in_atom = {atom: {"fit": [], "error": []} for atom in atom_list}
        fermi_expand_in_atom = {atom: {"fit": [], "error": []} for atom in atom_list}
        # pprint.pprint(A_expand_data[0]['hyperfine'])
        for atom in atom_list:
            fit_res = fitting(length_of_vectors, [obj['hyperfine_tensor'][str(atom)][0][1] for obj in A_expand_data], 1,
                              check=True)
            # fit_res = fitting(length_of_vectors, [obj['fermi_contact'][str(atom)] for obj in A_expand_data], 1,
            #                   check=True)
            # fit_res = fitting(length_of_vectors, [obj['hyperfine_v'][str(atom)] for obj in A_expand_data], 1,
            #                   check=True)
            # fit_res = fitting(length_of_vectors,
            #                   [obj['hyperfine_tensor'][str(atom)][col][row] if eig < 0 else
            #                    obj['hyperfine_eig'][str(atom)][eig]
            #                    for obj in A_expand_data], 1, check=True)
            # plot=f"expand_fit_check_{atom}"
            A_expand_in_atom[atom]["fit"] = fit_res["fit"][0]
            A_expand_in_atom[atom]["error"] = fit_res["error"][0]
            print(fit_res)

            # fit_res = fitting(length_of_vectors, [obj['fermi_contact'][str(atom)] for obj in A_expand_data], 1,
            #                   check=True)
            # # plot=f"expand_fit_check_{atom}"
            # fermi_expand_in_atom[atom]["fit"] = fit_res["fit"][0]
            # fermi_expand_in_atom[atom]["error"] = fit_res["error"][0]

            # paper plot
            if True:
                x = np.linspace(min(length_of_vectors), max(length_of_vectors), 100)
                # A_list = [
                #     obj['hyperfine_tensor'][str(atom)][col][row] if eig < 0 else obj['hyperfine_eig'][str(atom)][eig]
                #     for obj in A_expand_data]
                A_list = [obj['hyperfine_tensor'][str(atom)][0][1] for obj in A_expand_data]
                A_coef, A_error = np.polyfit(length_of_vectors, A_list, 1, cov=True)
                A_error = np.sqrt(np.diag(A_error))

                A_0 = np.poly1d(A_coef)(0)
                print("This is only a flag.", A_0)
                # fermi_list = [obj['fermi_contact'][str(atom)] for obj in A_expand_data]
                # fermi_list = [obj['fermi_contact'][str(atom)] for obj in A_expand_data]
                # fermi_coef = np.polyfit(length_of_vectors, fermi_list, 1)
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.scatter(length_of_vectors * 100, np.array(A_list) - A_0)
                ax1.plot(x * 100, np.poly1d(A_coef)(x) - A_0)

                # ax1.scatter(length_of_vectors, fermi_list)
                # ax1.plot(x, np.poly1d(fermi_coef)(x), label="Fermi contact")
                ax2 = ax1.twiny()
                ax3 = ax1.twiny()
                ax1.set_xlabel(r'$\frac{\delta L}{L_{\rm cls}}$ (%)', fontsize=15)
                ax1.set_ylabel(r"$\mathbf{A}_{\rm 0, xy}(V)-\mathbf{A}_{\rm 0, xy}(V_{\rm cls})$ (MHz)", fontsize=15)
                y_min, y_max = ax1.get_ylim()
                y_min -= 0.1 * (y_max - y_min)
                y_max += 0.1 * (y_max - y_min)
                interval = float("%.1g" % ((y_max - y_min) / 4))
                ax1.set_yticks(
                    np.arange(round(y_min / interval) * interval, round(y_max / interval) * interval + interval,
                              step=interval), fontsize=15)
                ax1.set_ylim([y_min, y_max])
                # 10.693999999999999, 10.826
                # 0, 1
                # print(ax1.get_xlim())

                # ax1.plot([0, 0], [y_min, y_max], "--")
                temp_list = [250 * i for i in range(9)[1:]]
                x_min, x_max = ax1.get_xlim()
                ax2.set_xticks(
                    (np.array([0] + [einstein_model(2)(t) * 100 for t in [1000, 2000]]) - x_min) / (x_max - x_min),
                    ["0", "1000", "2000"], fontsize=15)
                ax3.set_xticks(
                    (np.array([0] + [einstein_model(2)(t) * 100 for t in temp_list]) - x_min) / (x_max - x_min),
                    [])
                # ax1.set_xticks([0, 2e-3, 4e-3, 6e-3], ["0", "2", "4", "6"], fontsize=15)
                ax3.tick_params(direction="in")
                ax2.set_xlabel("$T$ (K)", fontsize=15)
                ax1.minorticks_on()
                ax1.xaxis.set_major_locator(MaxNLocator(5))
                ax1.yaxis.set_major_locator(MaxNLocator(5))
                ax1.tick_params(axis='both', which='major', labelsize=15)
                ax2.tick_params(axis='both', which='major', labelsize=15)
                fig.tight_layout()
                if dump_fig:
                    plt.savefig(work_dir + f"/lattice_fit_xy_{atom}.{fig_type}", dpi=600, format=fig_type)
                plt.close("all")
                if dump_json:
                    plot_data = {}
                    plot_data["plot1"] = {"type": "scatter",
                                          "x": list(length_of_vectors),
                                          "y": list(A_list)}
                    plot_data["plot2"] = {"type": "plot",
                                          "x": list(x),
                                          "y": list(np.poly1d(A_coef)(x))}
                    plot_data["xlabel"] = '$[a(T)/a(0)-1]$ (×10$^{-3}$)'
                    plot_data["ylabel"] = '$A$ (MHz)'
                    plot_data["xtick"] = {"origin": list(
                        (np.array([0] + [einstein_model(2)(t) for t in temp_list]) - x_min) / (x_max - x_min)),
                        "new": ["0", "", "", "", "1000", "", "", "", "2000"]}
                    plot_data["xlabel_top"] = '$T$ (K)'
                    json.dump(plot_data, open(f"fig2_{atom}.json", "w"), indent=4)

        pprint.pprint(A_expand_in_atom)

    if including_vib:
        # vib part
        # 某模式单个声子对应的温度 freq[nmode] / np.log(2) / K K 和振幅 np.sqrt(2 / freq[nmode] / C) amu^(1/2)A
        # 300K 对应的声子数 number_of_phonon_for_specfic_mode = 1 / (np.exp(freq[nmode] / (K * 300)) - 1)
        # 和振幅 np.sqrt(2 * number_of_phonon_for_specfic_mode / freq[nmode] / C) amu^(1/2)A
        mass, freq, mode_displace = json.load(open(work_dir + "/mode.json")).values()
        number_of_mode = len(freq)
        A_vib_data = json.load(open(work_dir + "/vib.json", "r"))

        A_vib_data_in_mode = {}
        for obj in A_vib_data:
            mode, displace = obj["path"].split("_")
            obj["mode"] = int(mode[1:])
            obj["displace"] = float(displace.split("/")[0])
            if obj["mode"] in A_vib_data_in_mode:
                A_vib_data_in_mode[obj["mode"]].append(obj)
            else:
                A_vib_data_in_mode[obj["mode"]] = [obj]

        ground_obj = A_vib_data_in_mode.pop(0)
        # ground_obj is a list, whose length == 1
        length_of_vectors = ground_obj[0]["length_of_vectors"]
        A_vib_in_atom = {atom: {"fit": [], "error": []} for atom in atom_list}
        A_vib_linear_in_atom = {atom: {"fit": [], "error": []} for atom in atom_list}
        x_avg = {"fit": [], "error": []}
        for mode in range(number_of_mode)[3:]:
            A_vib_data_in_mode[mode] += ground_obj
            A_vib_data_in_mode[mode].sort(key=lambda obj: obj["displace"])
            displace = [obj['displace'] for obj in A_vib_data_in_mode[mode]]
            energy = [obj['energy'] for obj in A_vib_data_in_mode[mode]]
            for atom in atom_list:
                hyperfine = [obj["fermi_contact"][str(atom)] for obj in A_vib_data_in_mode[mode]]
                # hyperfine = [obj["hyperfine"][str(atom)] for obj in A_vib_data_in_mode[mode]]
                # hyperfine = [
                #     obj['hyperfine_tensor'][str(atom)][col][row] if eig < 0 else obj['hyperfine_eig'][str(atom)][eig]
                #     for obj in A_vib_data_in_mode[mode]]
                # fix the first point in mode 27 and 252
                if mode in [27, 252]:
                    fit_res = fitting(displace[1:], hyperfine[1:], 2, check=True)
                else:
                    fit_res = fitting(displace, hyperfine, 2, check=True,
                                      plot=f"vib_fit_check_{atom}_{mode}" if (atom, mode) in [(1, 3),
                                                                                              (2, 3)] else False)

                # plot = f"vib_fit_check_{atom}_{mode}" if (atom, mode) == (1, 15) else False
                # [(1, 27), (2, 27), (153, 27), (27, 27), (232, 27), (232, 252), (231, 27)]
                # 二阶导和二次项系数差2
                # 在 0 处的二阶导, assuming A = a * x ** 2, A''|0 = 2 * a
                # np.polyval(np.polyder(np.polyder(total_A_coef)), 0)
                # np.polyval(np.polyder(np.polyder(Fermi_contact_term_coef)), 0)
                # for specific displacement, 2 * a =
                # (total_A_dict[nmode][displacement][atom]
                # + total_A_dict[nmode][displacement * -1][atom] - 2 * total_A0[atom])
                # / displacement ** 2
                A_vib_in_atom[atom]["fit"].append(fit_res["fit"][0])
                A_vib_in_atom[atom]["error"].append(fit_res["error"][0])
                A_vib_linear_in_atom[atom]["fit"].append(fit_res["fit"][1])
                A_vib_linear_in_atom[atom]["error"].append(fit_res["error"][1])

            # coef, cov = np.polyfit(displace, energy, 3, cov=True)
            energy = np.array(energy) - min(energy)
            coef, cov = curve_fit(lambda x, a, b: a * x ** 3 + b * x ** 2, displace, energy, bounds=([-1, 0], [1, 4]))
            cov = np.sqrt(np.diag(cov))
            # x_avg["fit"].append(-3 * coef[0] * freq[mode] / (4 * coef[1] ** 2))
            x_avg["fit"].append(-3 * coef[0] / (freq[mode] ** 3 * C ** 2))
            x_avg["error"].append(-3 * cov[0] / (freq[mode] ** 3 * C ** 2))
            # ratio = cov[0] / coef[0]
            # if abs(ratio) > 1e8:
            #     print(f'mode: {mode} x_avg: {x_avg["fit"][-1]} ± {x_avg["error"][-1]} ratio: {ratio}')
            if False:
                # 测试有效质量是否收敛
                energy = np.array(energy) - min(energy)
                coef = np.polyfit(displace, energy, 2)
                x = np.linspace(displace[0], displace[-1], 100)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(displace, energy, color="red")
                ax.plot(x, np.poly1d(coef)(x), "r--", label="fitting")
                ax.plot(x, (freq[mode] * x) ** 2 / 2 * C, label="harmonic")

                # +声子波函数
                number_of_phonon = 0
                while True:
                    if freq[mode] * (number_of_phonon + 0.5) > np.poly1d(coef)(x[-1]):
                        break
                    else:
                        ax.plot([x[0], x[-1]],
                                [freq[mode] * (number_of_phonon + 0.5),
                                 freq[mode] * (number_of_phonon + 0.5)], "--", linewidth=0.7)
                        x2 = x * (freq[mode] * C) ** 0.5
                        phonon_wf = np.e ** (-x2 ** 2 / 2) * np.polynomial.hermite.hermval(x2,
                                                                                           [0 if _ < number_of_phonon
                                                                                            else 1 for _ in range(
                                                                                               number_of_phonon + 1)])
                        # 归一化, 0.25 跟据显示效果可以调整
                        ax.plot(x,
                                .25 * freq[mode] / (sum(phonon_wf ** 2) * (x[-1] - x[0]) / len(x)) ** 0.5 * phonon_wf \
                                + freq[mode] * (number_of_phonon + 0.5), linewidth=0.7)
                        number_of_phonon += 1

                ax.set_xlabel(r'$X$ ($\rm \sqrt{amu}\cdot\AA$)', fontsize=15)
                ax.set_ylabel("Energy (eV)", fontsize=15)
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.minorticks_on()
                ax.yaxis.set_major_locator(MaxNLocator(5))
                plt.legend(fontsize=15)
                plt.tight_layout()
                plt.savefig(work_dir + f"/mass_check_{mode}.pdf", dpi=600, format="pdf")
                plt.close("all")

        # fitting error at 300 K
        if True:
            # t[289] == 300
            t_range = np.array(range(501)[11:])
            number_of_phonon_per_mode = [1 / (np.exp(np.array(freq[3:]) / (K * t)) - 1) for t in t_range]
            fill_between = True
            x_avg["fit"] = np.array(x_avg["fit"])
            x_avg["error"] = np.array(x_avg["error"])
            for atom in atom_list:
                print(f"atom: {atom}")
                A_vib = {}
                dA_vib_dT = {}
                d2A_vib_dT2 = {}
                # 125 is from anharmonic vibration ax3/<x> and linear x dependent A(x)=ax
                A_vib_125 = {}
                dA_vib_dT_125 = {}
                print(A_vib_in_atom[atom]["fit"])
                for flag in ["fit", "error"]:
                    A_vib_in_atom[atom][flag] = np.array(A_vib_in_atom[atom][flag])
                    A_vib_linear_in_atom[atom][flag] = np.array(A_vib_linear_in_atom[atom][flag])
                    # 1 / (E ** (f / k / T) - 1) * A / f / c
                    A_vib[flag] = [A_vib_in_atom[atom][flag] * pn / freq[3:] / C for pn in number_of_phonon_per_mode]
                    # exp(f / (T * k)) / (T ** 2 * k * (exp(f / (T * k)) - 1) ** 2) * A / c
                    dA_vib_dT[flag] = [A_vib_in_atom[atom][flag] * (pn + 1) * pn / (K * t ** 2 * C) for t, pn in
                                       zip(t_range, number_of_phonon_per_mode)]
                    d2A_vib_dT2[flag] = [A_vib_in_atom[atom][flag] * (
                            -2 - freq[3:] / (t * K) + 2 * np.array(freq[3:]) * np.exp(freq[3:] / (t * K)) / (
                            t * K * (np.exp(freq[3:] / (t * K)) - 1))) * np.exp(freq[3:] / (t * K)) / (
                                                 t ** 3 * K * (np.exp(freq[3:] / (t * K)) - 1) ** 2) / C for t, pn
                                         in zip(t_range, number_of_phonon_per_mode)]

                A_vib_125["fit"] = [A_vib_linear_in_atom[atom]["fit"] * x_avg["fit"] * pn for pn in
                                    number_of_phonon_per_mode]
                A_vib_125["error"] = [((A_vib_linear_in_atom[atom]["fit"] * x_avg["error"]) ** 2 + (
                        A_vib_linear_in_atom[atom]["error"] * x_avg["fit"]) ** 2) ** .5 * pn for pn in
                                      number_of_phonon_per_mode]
                dA_vib_dT_125["fit"] = [A_vib_linear_in_atom[atom]["fit"] *
                                        x_avg["fit"] * (pn + 1) * pn * freq[3:] / (K * t ** 2) for t, pn in
                                        zip(t_range, number_of_phonon_per_mode)]
                dA_vib_dT_125["error"] = [((A_vib_linear_in_atom[atom]["fit"] * x_avg["error"]) ** 2 +
                                           (A_vib_linear_in_atom[atom]["error"] * x_avg["fit"]) ** 2) ** .5 *
                                          (pn + 1) * pn * freq[3:] / (K * t ** 2) for t, pn in
                                          zip(t_range, number_of_phonon_per_mode)]
                print(f"zero point vib: {sum(A_vib_in_atom[atom]['fit'] * .5 / freq[3:] / C)} ± "
                      f"{sum(A_vib_in_atom[atom]['error'] * .5 / freq[3:] / C)}")
                print(f"zero point 125: {sum(A_vib_linear_in_atom[atom]['fit'] * x_avg['fit'] * .5)} ± "
                      f"{sum(((A_vib_linear_in_atom[atom]['fit'] * x_avg['error']) ** 2 + (A_vib_linear_in_atom[atom]['error'] * x_avg['fit']) ** 2) ** .5 * .5)}")

                Ci = A_vib_in_atom[atom]["fit"] / freq[3:] / C
                Ci_error = A_vib_in_atom[atom]["error"] / freq[3:] / C
                Ci_125 = A_vib_linear_in_atom[atom]["fit"] * x_avg["fit"]
                Ci_125_error = ((A_vib_linear_in_atom[atom]["fit"] * x_avg["error"]) ** 2 +
                                (A_vib_linear_in_atom[atom]["error"] * x_avg["fit"]) ** 2) ** .5
                step = 1e-3
                start = freq[3]

                f_colle = []

                temp_c = 0
                temp_c_err = 0
                c_colle = []
                c_err_colle = []

                temp_c_125 = 0
                temp_c_125_err = 0
                c_125_colle = []
                c_125_err_colle = []

                for f, c, c_err, c_125, c_125_err in zip(freq[3:], Ci, Ci_error, Ci_125, Ci_125_error):
                    if f > start + step:
                        f_colle.append(start + step / 2)
                        start = f
                        c_colle.append(temp_c)
                        c_err_colle.append(temp_c_err ** .5)
                        temp_c = 0
                        temp_c_err = 0

                        c_125_colle.append(temp_c_125)
                        c_125_err_colle.append(temp_c_125_err ** .5)
                        temp_c_125 = 0
                        temp_c_125_err = 0
                    else:
                        temp_c += c
                        temp_c_err += c_err ** 2

                        temp_c_125 += c_125
                        temp_c_125_err += c_125_err ** 2

                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                A_vib_fit = np.array([sum(i) for i in A_vib["fit"]])
                A_vib_error = np.array([np.linalg.norm(i, 2) for i in A_vib["error"]])
                ax1.plot(t_range, A_vib_fit, "g", label="harmonic")
                A_vib_125_fit = np.array([sum(i) for i in A_vib_125["fit"]])
                A_vib_125_error = np.array([np.linalg.norm(i, 2) for i in A_vib_125["error"]])
                ax1.plot(t_range, A_vib_125_fit, "orange", label="anharmonic")
                if dump_json:
                    plot_data = {}
                    plot_data["plot1"] = {"type": "plot",
                                          "x": t_range.tolist(),
                                          "y": list(A_vib_fit),
                                          "color": "green",
                                          "label": "dynamic"}
                if fill_between:
                    ax1.fill_between(t_range, A_vib_fit + A_vib_error, A_vib_fit - A_vib_error,
                                     facecolor="b", alpha=0.5)
                    ax1.fill_between(t_range, A_vib_125_fit + A_vib_125_error, A_vib_125_fit - A_vib_125_error,
                                     facecolor="b", alpha=0.5)
                if including_expand:
                    A_expand_fit = A_expand_in_atom[atom]["fit"] * einstein_model(2)(t_range)
                    print("zero point volume:", A_expand_in_atom[atom]["fit"] * 0.004)
                    A_expand_error = A_expand_in_atom[atom]["error"] * einstein_model(2)(t_range)
                    print("delta A", A_expand_fit[289], A_vib_fit[289], A_expand_fit[289] + A_vib_fit[289])
                    ax1.plot(t_range, A_expand_fit, "b", label="static")
                    ax1.plot(t_range, A_expand_fit + A_vib_fit + A_vib_125_fit, "r", label="total")
                    if dump_json:
                        plot_data["plot2"] = {"type": "plot",
                                              "x": t_range.tolist(),
                                              "y": list(A_expand_fit),
                                              "color": "blue",
                                              "label": "static"}
                        plot_data["plot3"] = {"type": "plot",
                                              "x": t_range.tolist(),
                                              "y": list(A_expand_fit + A_vib_fit),
                                              "color": "red",
                                              "label": "total"}
                    if fill_between:
                        ax1.fill_between(t_range, A_expand_fit + A_expand_error, A_expand_fit - A_expand_error,
                                         facecolor="b", alpha=0.5)
                        ax1.fill_between(t_range,
                                         A_expand_fit + A_vib_fit + A_vib_125_fit +
                                         (A_expand_error ** 2 + A_vib_error ** 2 + A_vib_125_error ** 2) ** .5,
                                         A_expand_fit + A_vib_fit + A_vib_125_fit -
                                         (A_expand_error ** 2 + A_vib_error ** 2 + A_vib_125_error ** 2) ** .5,
                                         facecolor="b", alpha=0.5)
                ax1.set_xlabel('$T$ (K)', fontsize=15)
                ax1.set_ylabel(r'$\delta$A (MHz)', fontsize=15)
                if dump_json:
                    plot_data["xlabel"] = "$T$ (K)"
                    plot_data["ylabel"] = r'$\delta$A (MHz)'
                    json.dump(plot_data, open(f"fig2_{atom}.json", "w"), indent=4)
                ax1.legend(fontsize=15)
                ax1.minorticks_on()
                ax1.xaxis.set_major_locator(MaxNLocator(5))
                ax1.yaxis.set_major_locator(MaxNLocator(5))
                ax1.tick_params(axis='both', which='major', labelsize=15)
                fig.tight_layout()
                if dump_fig:
                    plt.savefig(work_dir + f"/at_{atom}.{fig_type}", dpi=600, format=fig_type)
                plt.show()
                plt.close("all")

                fig = plt.figure()
                ax2 = fig.add_subplot(111)
                dA_vib_dT_fit = np.array([sum(i) for i in dA_vib_dT["fit"]])
                dA_vib_dT_error = np.array([np.linalg.norm(i, 2) for i in dA_vib_dT["error"]])

                d2A_vib_dT2_fit = np.array([sum(i) for i in d2A_vib_dT2["fit"]])
                d2A_vib_dT2_error = np.array([np.linalg.norm(i, 2) for i in d2A_vib_dT2["error"]])

                ax2.plot(t_range, 1e6 * dA_vib_dT_fit, "g", label="harmonic")
                dA_vib_dT_125_fit = np.array([sum(i) for i in dA_vib_dT_125["fit"]])
                dA_vib_dT_125_error = np.array([np.linalg.norm(i, 2) for i in dA_vib_dT_125["error"]])
                ax2.plot(t_range, 1e6 * dA_vib_dT_125_fit, "orange", label="anharmonic")
                if fill_between:
                    ax2.fill_between(t_range, 1e6 * (dA_vib_dT_fit + dA_vib_dT_error),
                                     1e6 * (dA_vib_dT_fit - dA_vib_dT_error),
                                     facecolor="b", alpha=0.5)
                    ax2.fill_between(t_range, 1e6 * (dA_vib_dT_125_fit + dA_vib_dT_125_error),
                                     1e6 * (dA_vib_dT_125_fit - dA_vib_dT_125_error),
                                     facecolor="b", alpha=0.5)

                if including_expand:
                    dA_expand_dT_fit = A_expand_in_atom[atom]["fit"] * einstein_model(1)(t_range)
                    dA_expand_dT_error = A_expand_in_atom[atom]["error"] * einstein_model(1)(t_range)
                    # d2A_expand_dT2_fit = A_expand_in_atom[atom]["fit"] * einstein_model(3)(t_range)
                    # d2A_expand_dT2_error = A_expand_in_atom[atom]["error"] * einstein_model(3)(t_range)
                    ax2.plot(t_range, 1e6 * dA_expand_dT_fit, "b", label="static")
                    ax2.plot(t_range, 1e6 * (dA_expand_dT_fit + dA_vib_dT_fit + dA_vib_dT_125_fit), "r", label="total")
                    print(r"dA/dT", dA_expand_dT_fit[289], dA_vib_dT_fit[289],
                          dA_expand_dT_fit[289] + dA_vib_dT_fit[289])
                    # print(r"d2A/dT2", d2A_vib_dT2_fit[289] + d2A_expand_dT2_fit[289],
                    #       d2A_vib_dT2_error[289] + d2A_expand_dT2_error[289])
                    print(r"d2A/dT2", d2A_vib_dT2_fit[289], d2A_vib_dT2_error[289])
                    if fill_between:
                        ax2.fill_between(t_range, 1e6 * (dA_expand_dT_fit + dA_expand_dT_error),
                                         1e6 * (dA_expand_dT_fit - dA_expand_dT_error),
                                         facecolor="b", alpha=0.5)
                        ax2.fill_between(t_range,
                                         1e6 * (dA_expand_dT_fit + dA_vib_dT_fit + dA_vib_dT_125_fit +
                                                (
                                                        dA_expand_dT_error ** 2 + dA_vib_dT_error ** 2 + dA_vib_dT_125_error ** 2) ** .5),
                                         1e6 * (dA_expand_dT_fit + dA_vib_dT_fit + dA_vib_dT_125_fit -
                                                (
                                                        dA_expand_dT_error ** 2 + dA_vib_dT_error ** 2 + dA_vib_dT_125_error ** 2) ** .5),
                                         facecolor="b", alpha=0.5)

                # if atom_name[atom_list.index(atom)] in exp:
                #     ax2.scatter(*exp[atom_name[atom_list.index(atom)]],marker="o")
                ax2.set_xlabel('$T$ (K)', fontsize=15)
                ax2.set_ylabel(r'$\frac{dA}{dT}$ (Hz/K)', fontsize=15)
                ax2.legend(fontsize=15)
                ax2.minorticks_on()
                ax2.tick_params(axis='both', which='major', labelsize=15)
                ax2.xaxis.set_major_locator(MaxNLocator(5))
                ax2.yaxis.set_major_locator(MaxNLocator(5))
                fig.tight_layout()
                if dump_fig:
                    plt.savefig(work_dir + f"/att_{atom}.{fig_type}", dpi=600, format=fig_type)
                plt.close("all")

                fig = plt.figure()
                ax3 = fig.add_subplot(111)
                ax3.bar(f_colle, c_colle, width=step * 1.2)
                ax3.bar(f_colle, c_err_colle, width=step * 1.2)
                ax3.set_xlim(0.02, 0.18)
                ax3.plot(ax3.get_xlim(), [0, 0], "k--")
                # ax5 = ax3.twinx()
                freq_set = np.linspace(freq[3], freq[-1], 500)
                number_of_phonon_per_mode_300 = 1 / (np.exp(np.array(freq_set) / (K * 289)) - 1)
                number_of_phonon_per_mode_500 = 1 / (np.exp(np.array(freq_set) / (K * 489)) - 1)
                # plot1 = ax5.plot(freq_set, number_of_phonon_per_mode_300, "r", label="300 K")
                # plot3 = ax5.plot(freq_set, number_of_phonon_per_mode_500, "g", label="500 K")
                ax3_ymin, ax3_ymax = min(c_colle), max(c_colle)
                ax3.set_ylim([ax3_ymin - .01 * (ax3_ymax - ax3_ymin), ax3_ymax + .01 * (ax3_ymax - ax3_ymin)])
                ratio = (ax3_ymax + .01 * (ax3_ymax - ax3_ymin)) / (ax3_ymin - .01 * (ax3_ymax - ax3_ymin))
                # ax5.axes.set_yticks(np.arange(0, 1, step=0.2), fontsize=15)
                # ax5.set_ylim([number_of_phonon_per_mode[489][0] * 1.01 / ratio,
                #               number_of_phonon_per_mode[489][0] * 1.01])

                ax6 = ax3.twinx()
                phonon_freq, phonon_dos = read_dos(work_dir + "/total_dos.dat").values()
                plot2 = ax6.plot(phonon_freq, phonon_dos, "k", label="phonon DOS")
                ax6_ymax = max(phonon_dos)
                ax6.set_ylim([ax6_ymax * 1.01 / ratio, ax6_ymax * 1.01])

                ax3.yaxis.set_ticks_position('right')
                ax3.yaxis.set_label_position('right')
                ax3.minorticks_on()
                ax3.set_xlabel('$\hbar\omega$ (eV)', fontsize=15)
                ax3.set_ylabel('$c_i$ (MHz)', fontsize=15)
                ax3.tick_params(axis='both', which='major', labelsize=15)
                # ax5.yaxis.set_ticks_position('left')
                # ax5.yaxis.set_label_position('left')
                # ax5.minorticks_on()
                # ax5.set_ylabel(r'$\langle n\rangle$', fontsize=15)
                # ax5.tick_params(axis='both', which='major', labelsize=15)
                ax3.xaxis.set_major_locator(MaxNLocator(5))
                ax3.yaxis.set_major_locator(MaxNLocator(5))
                # ax5.yaxis.set_major_locator(MaxNLocator(5))

                ax6.axes.yaxis.set_ticklabels([])
                ax6.axes.yaxis.set_ticks([])
                # plots = plot1 + plot2
                # plots = plot1 + plot3
                # labels = [aplot.get_label() for aplot in plots]
                # ax3.legend(plots, labels)
                plt.legend()

                fig.tight_layout()
                if dump_json:
                    plot_data = {}
                    plot_data["plot1"] = {"type": "bar",
                                          "x": list(f_colle),
                                          "y": list(c_colle),
                                          "step": step}
                    plot_data["plot2"] = {"type": "plot",
                                          "x": list(ax3.get_xlim()),
                                          "y": [0, 0],
                                          "color": "black",
                                          "lineshape": "dashed"}
                    plot_data["plot3"] = {"type": "plot",
                                          "x": list(freq_set),
                                          "y": list(number_of_phonon_per_mode_300),
                                          "color": "red",
                                          "label": "300 K"}
                    plot_data["xlabel"] = '$\hbar\omega$ (eV)'
                    plot_data["ylabel"] = '$c_i$ (MHz)'
                    plot_data["ylabel_right"] = r'$ \bar n$'
                    json.dump(plot_data, open(f"fig3_{atom}.json", "w"), indent=4)
                if dump_fig:
                    plt.savefig(work_dir + f"/A_phonon_{atom}.{fig_type}", dpi=600, format=fig_type)
                plt.close("all")

                fig = plt.figure()
                ax3 = fig.add_subplot(111)

                ax3.bar(f_colle, np.array(c_125_colle) * 1e3, width=step * 1.2)
                ax3.bar(f_colle, np.array(c_125_err_colle) * 1e3, width=step * 1.2)
                ax3.set_xlim(0.02, 0.18)
                ax3.plot(ax3.get_xlim(), [0, 0], "k--")

                ax5 = ax3.twinx()
                freq_set = np.linspace(freq[3], freq[-1], 500)
                number_of_phonon_per_mode_300 = 1 / (np.exp(np.array(freq_set) / (K * 289)) - 1)
                ax5.plot(freq_set, number_of_phonon_per_mode_300, "r", label="phonon number (300 K)")
                ax3_ymin, ax3_ymax = min(c_125_colle) * 1e3, max(c_125_colle) * 1e3
                ax3.set_ylim([ax3_ymin - .01 * (ax3_ymax - ax3_ymin), ax3_ymax + .01 * (ax3_ymax - ax3_ymin)])
                ratio = (ax3_ymax + .01 * (ax3_ymax - ax3_ymin)) / (ax3_ymin - .01 * (ax3_ymax - ax3_ymin))
                ax5.axes.set_yticks(np.arange(0, .5, step=0.1), fontsize=15)
                ax5.set_ylim([number_of_phonon_per_mode[289][0] * 1.01 / ratio,
                              number_of_phonon_per_mode[289][0] * 1.01])
                ax5.yaxis.set_ticks_position('left')
                ax5.yaxis.set_label_position('left')
                ax5.set_ylabel(r'$\langle n\rangle$', fontsize=15)
                ax5.tick_params(axis='both', which='major', labelsize=15)
                ax3.yaxis.set_ticks_position('right')
                ax3.yaxis.set_label_position('right')
                ax3.minorticks_on()
                ax3.set_xlabel('$\hbar\omega$ (eV)', fontsize=15)
                ax3.set_ylabel('$b_i$ (kHz)', fontsize=15)
                ax3.tick_params(axis='both', which='major', labelsize=15)
                ax3.xaxis.set_major_locator(MaxNLocator(5))
                ax3.yaxis.set_major_locator(MaxNLocator(5))
                fig.tight_layout()

                if dump_fig:
                    plt.savefig(work_dir + f"/A_phonon_anh_{atom}.{fig_type}", dpi=600, format=fig_type)
                plt.close("all")

                fig = plt.figure()
                # ax3 = fig.add_subplot(121)
                ax4 = fig.add_subplot(111)
                # ax5 = ax4.twinx()
                # plots = ax5.plot(range(number_of_mode)[3:], np.array(freq[3:]) * 1000, label="$E_{ph}$", color="red")
                for flag in ["fit", "error"]:
                    derivative_300 = dA_vib_dT[flag][289]
                    derivative_300 *= 1e6
                    # ax3.plot(range(number_of_mode)[3:], derivative_300, label=flag)
                    # if flag == "fit":
                    #     ax3_ymin = min(derivative_300)
                    #     ax3_ymax = max(derivative_300)
                    #     ax3.set_ylim([ax3_ymin, ax3_ymax])
                    #     ax5 = ax3.twinx()
                    #     ax5.plot(range(number_of_mode)[3:], number_of_phonon_per_mode[289], "r", label="300 K")
                    #     ax5.set_ylim([number_of_phonon_per_mode[289][0] / ax3_ymax * ax3_ymin,
                    #                   number_of_phonon_per_mode[289][0]])

                    total_derivative_300 = sum(derivative_300)
                    print(f"300 K dA/dT {flag}: {total_derivative_300}")
                    for mode, der in sorted(list(enumerate(derivative_300)), key=lambda x: np.abs(x[1]),
                                            reverse=True)[:10]:
                        # 因为freq[3:], 现在phonon mode 从3开始, +3转化为从0开始
                        print(f"phonon contribution: ({atom},{mode + 3}) {der} {der / total_derivative_300}")

                    accumulate_list = [0]
                    ph_energy = [freq[3]]
                    for der, f in zip(derivative_300, freq[3:]):
                        if f > ph_energy[-1]:
                            while f > ph_energy[-1]:
                                ph_energy.append(ph_energy[-1] + step)
                                accumulate_list.append(accumulate_list[-1])
                        else:
                            ph_energy.append(ph_energy[-1] + step)
                            accumulate_list.append(accumulate_list[-1] +
                                                   (der if flag == "fit" else der ** 2))
                    if flag == "error":
                        accumulate_list = np.power(np.array(accumulate_list), .5)

                    ax4.plot(ph_energy, accumulate_list, label=flag)

                # ax3.set_xlabel("mode")
                # ax3.set_ylabel(r'$\frac{dA}{dT}$ (Hz/K)')

                # labels = [aplot.get_label() for aplot in plots]
                # ax4.legend(plots, labels, fontsize=15)
                # ax5.set_ylabel("phonon energy (meV)", fontsize=15)
                # ax5.yaxis.set_ticks_position('right')
                # ax5.yaxis.set_label_position('right')
                ax4.tick_params(axis='both', which='major', labelsize=15)
                # ax5.tick_params(axis='both', which='major', labelsize=15)
                # ax5.minorticks_on()
                ax4.minorticks_on()
                ax4.xaxis.set_major_locator(MaxNLocator(5))
                ax4.yaxis.set_major_locator(MaxNLocator(5))
                # ax5.yaxis.set_major_locator(MaxNLocator(5))
                ax4.set_xlabel(r"$\hbar\omega$ (eV)", fontsize=15)
                ax4.set_ylabel(r'$\frac{dA}{dT}_{vib}$ (Hz/K)', fontsize=15)
                # ax3.legend()
                ax4.legend(fontsize=15)
                # ax5.legend()
                fig.tight_layout()
                plt.savefig(work_dir + f"/fit_result_and_error_300_{atom}.png", dpi=300)
                plt.close("all")
