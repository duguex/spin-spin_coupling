# -*- coding: utf-8 -*-
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MaxNLocator
from sympy import *

plt.rcParams["font.size"] = 15


def read_elastic_tensor(csv_path, json_path):
    with open(csv_path, "r") as f:
        elastic_tensor = []
        elastic_tensor_dict = {}
        counter = 0
        for line in f.readlines():
            element_list = line.strip().split(",")
            scale = float(element_list[0])
            element_list = list(map(float, element_list[1:]))
            elastic_tensor.append(element_list)
            counter += 1
            if counter >= 6:
                counter = 0
                elastic_tensor_dict[scale] = elastic_tensor
                elastic_tensor = []

    json.dump(elastic_tensor_dict, open(json_path, "w"), indent=4)


def independent_element_c3v(elastic_tensor):
    # elaster_tensor = [['10687.4749', '1556.7189', '1556.7189', '-2.3041', '-2.4087', '-2.3041'],
    #                   ['1556.7189', '10687.4749', '1556.7189', '-2.3041', '-2.3041', '-2.4087'],
    #                   ['1556.7189', '1556.7189', '10687.4749', '-2.4087', '-2.3041', '-2.3041'],
    #                   ['-2.3041', '-2.3041', '-2.4087', '5905.1544', '-4.1927', '-4.1927'],
    #                   ['-2.4087', '-2.3041', '-2.3041', '-4.1927', '5905.1544', '-4.1927'],
    #                   ['-2.3041', '-2.4087', '-2.3041', '-4.1927', '-4.1927', '5905.1544']]
    elastic_tensor = np.array(elastic_tensor)
    # c3v check
    element_0_0 = elastic_tensor[0][0]
    element_0_1 = elastic_tensor[0][1]
    element_3_3 = elastic_tensor[3][3]
    element_3_4 = elastic_tensor[3][4]
    element_0_3 = elastic_tensor[0][3]
    element_0_4 = elastic_tensor[0][4]

    for i in range(3):
        for j in range(3):
            if i == j:
                assert np.isclose(element_0_0, elastic_tensor[i][j]), \
                    f"C3v symmetry is not satisfied for element {i}, {j}."
            else:
                assert np.isclose(element_0_1, elastic_tensor[i][j]), \
                    f"C3v symmetry is not satisfied for element {i}, {j}."

    for i in range(3, 6):
        for j in range(3, 6):
            if i == j:
                assert np.isclose(element_3_3, elastic_tensor[i][j]), \
                    f"C3v symmetry is not satisfied for element {i}, {j}."
            else:
                assert np.isclose(element_3_4, elastic_tensor[i][j]), \
                    f"C3v symmetry is not satisfied for element {i}, {j}."

    for i in range(3):
        for j in range(3, 6):
            if (j - i - 1) % 3 == 0:
                assert np.isclose(element_0_4, elastic_tensor[i][j]), \
                    f"C3v symmetry is not satisfied for element {i}, {j}."
            else:
                assert np.isclose(element_0_3, elastic_tensor[i][j]), \
                    f"C3v symmetry is not satisfied for element {i}, {j}."

    for i in range(3, 6):
        for j in range(3):
            if (i - j - 1) % 3 == 0:
                assert np.isclose(element_0_4, elastic_tensor[i][j]), \
                    f"C3v symmetry is not satisfied for element {i}, {j}."
            else:
                assert np.isclose(element_0_3, elastic_tensor[i][j]), \
                    f"C3v symmetry is not satisfied for element {i}, {j}."
    return [element_0_0, element_0_1, element_3_3, element_3_4, element_0_4, element_0_3]


def construct_tensor_c3v(elements):
    # elements in order of [element_0_0, element_0_1, element_3_3, element_3_4, element_0_4, element_0_3]
    tensor = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            tensor[i][j] = elements[0] if i == j else elements[1]

    for i in range(3, 6):
        for j in range(3, 6):
            tensor[i][j] = elements[2] if i == j else elements[3]

    for i in range(3):
        for j in range(3, 6):
            tensor[i][j] = elements[4] if (j - i - 1) % 3 == 0 else elements[5]

    for i in range(3, 6):
        for j in range(3):
            tensor[i][j] = elements[4] if (i - j - 1) % 3 == 0 else elements[5]

    return tensor


def fitting_elastic_tensor():
    elastic_tensor_dict = json.load(open(work_dir + "/elastic_tensor_nv_diamond_444.json", "r"))
    scale_list = list(elastic_tensor_dict.keys())
    scale_list = list(map(float, scale_list))
    scale_list = np.array(scale_list) - 1
    argsort = scale_list.argsort()
    scale_list = scale_list[argsort]

    tensor_list = list(elastic_tensor_dict.values())
    tensor_list = list(map(independent_element_c3v, tensor_list))
    tensor_list = np.array(tensor_list)
    assert tensor_list.shape[1] == 6, "wrong shape for tensor"
    tensor_list = tensor_list[argsort].T

    coef_list = []
    error_list = []

    scale_sample = np.linspace(scale_list[0], scale_list[-1], 100)
    fig = plt.figure()

    for ax_index, tensor_index in zip(range(321, 327), range(6)):
        coef, cov = np.polyfit(scale_list, tensor_list[tensor_index], 1, cov=True)
        coef_list.append(coef.tolist())
        error_list.append(np.sqrt(cov.diagonal()).tolist())

        ax = fig.add_subplot(ax_index)
        ax.scatter(scale_list, tensor_list[tensor_index], marker="x", c="r")
        ax.plot(scale_sample, np.poly1d(coef)(scale_sample))
        ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
        ax.set_title(title_list[tensor_index])
    plt.subplots_adjust(hspace=0.8)
    plt.show()

    json.dump([coef_list, error_list], open(work_dir + "/fitting_elastic_tensor_nv_diamond_444.json", "w"))


def get_elastic_tensor(scale):
    coef_list, error_list = json.load(open(work_dir + "/fitting_elastic_tensor_nv_diamond_444.json", "r"))
    elements = [np.poly1d(coef)(scale) for coef in coef_list]
    errors = [((error[0] * scale) ** 2 + error[1] ** 2) ** .5 for error in error_list]
    np.set_printoptions(suppress=True)
    print(elements)
    print(errors)
    return construct_tensor_c3v(elements), construct_tensor_c3v(errors)


def get_elastic_coef(scale):
    _elastic_tensor, error = get_elastic_tensor(scale)
    return sum(_elastic_tensor[0][:3]), np.linalg.norm(error[0][:3]),


def stress_ellipsoid(normal=True, atom=0):
    # u from 0 to 360
    # v from 0 to 180
    u = np.linspace(0, 2 * np.pi, 3000)
    v = np.linspace(0, np.pi, 3000)

    # 单位应变/向量
    # strain = np.array([np.outer(np.cos(u), np.sin(v)).reshape(u.shape[0] * v.shape[0]),
    #                    np.outer(np.sin(u), np.sin(v)).reshape(u.shape[0] * v.shape[0]),
    #                    np.outer(np.ones(np.size(u)), np.cos(v)).reshape(u.shape[0] * v.shape[0])])
    stress = np.array([np.outer(np.cos(u), np.sin(v)).reshape(u.shape[0] * v.shape[0]),
                       np.outer(np.sin(u), np.sin(v)).reshape(u.shape[0] * v.shape[0]),
                       np.outer(np.ones(np.size(u)), np.cos(v)).reshape(u.shape[0] * v.shape[0])])

    if normal:
        temp_elastic_moduli = elastic_moduli_inv_in_kBar[:3, :3]
        temp_hyper_strain = hyper_strain[:3, atom].T
    else:
        temp_elastic_moduli = elastic_moduli_inv_in_kBar[3:, 3:]
        temp_hyper_strain = hyper_strain[3:, atom].T

    # stress = temp_elastic_moduli.dot(strain)
    strain = temp_elastic_moduli.dot(stress)
    # hyperfine to heatmap
    color_data = temp_hyper_strain.dot(strain)
    max = color_data.max()
    # max_strain = strain.T[np.where(color_data == max)[0][0]]
    # max_stress = temp_elastic_moduli.dot(max_strain.T)
    max_stress = stress.T[np.where(color_data == max)[0][0]]
    print(normal, atom, round(max, 2),
          "&(" + ",".join([str(round(i, 3)) for i in (max_stress / np.linalg.norm(max_stress))]) + ")")
    min = color_data.min()
    # min_strain = strain.T[np.where(color_data == min)[0][0]]
    # min_stress = temp_elastic_moduli.dot(min_strain.T)
    # print(normal, atom, "min", min, *min_stress)
    color_data = color_data.reshape((u.shape[0], v.shape[0]))
    norm = colors.Normalize(vmin=min, vmax=max)

    if True:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(stress[0].reshape((u.shape[0], v.shape[0])), stress[1].reshape((u.shape[0], v.shape[0])),
        #                 stress[2].reshape((u.shape[0], v.shape[0])), facecolors=plt.cm.jet(norm(color_data)))
        ax.plot_surface(strain[0].reshape((u.shape[0], v.shape[0])), strain[1].reshape((u.shape[0], v.shape[0])),
                        strain[2].reshape((u.shape[0], v.shape[0])), facecolors=plt.cm.jet(norm(color_data)))
        cbar = fig.colorbar(cm.ScalarMappable(cmap=plt.cm.jet, norm=norm), ax=ax, label="$A$ (kHz/kbar)",
                            shrink=0.75)
        cbar.ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
        # ticks=[round(min, 1), 0, round(max, 1)],
        # ax.contourf(x, y, z, zdir='x', offset=-2 * a, cmap=cm.coolwarm)
        # ax.contourf(x, y, z, zdir='y', offset=1.8 * b, cmap=cm.coolwarm)
        # ax.contourf(x, y, z, zdir='z', offset=-2 * c, cmap=cm.coolwarm)

        # if normal:
        #     ax.set_xlabel('$\sigma_{XX}$ (kbar)')
        #     ax.set_ylabel('$\sigma_{YY}$ (kbar)')
        #     ax.set_zlabel('$\sigma_{ZZ}$ (kbar)')
        # else:
        #     ax.set_xlabel('$\sigma_{YZ}$ (kbar)')
        #     ax.set_ylabel('$\sigma_{ZX}$ (kbar)')
        #     ax.set_zlabel('$\sigma_{XY}$ (kbar)')
        if normal:
            ax.set_xlabel('$\epsilon_{XX}$ (kbar$^{-1}$)')
            ax.set_ylabel('$\epsilon_{YY}$ (kbar$^{-1}$)')
            ax.set_zlabel('$\epsilon_{ZZ}$ (kbar$^{-1}$)')
        else:
            ax.set_xlabel('$\epsilon_{YZ}$ (kbar$^{-1}$)')
            ax.set_ylabel('$\epsilon_{ZX}$ (kbar$^{-1}$)')
            ax.set_zlabel('$\epsilon_{XY}$ (kbar$^{-1}$)')

        ax.xaxis.set_major_locator(MaxNLocator(2))
        ax.yaxis.set_major_locator(MaxNLocator(2))
        ax.zaxis.set_major_locator(MaxNLocator(2))
        ax.ticklabel_format(style='sci', scilimits=(0, 2), axis='both')
        ax.view_init(22.5, -45)
        plt.tight_layout()
        plt.savefig(f"{'normal' if normal else 'shear'}_{atom}.png")
        plt.close("all")
        # plt.show()

def fit_dA_depslion():
    import os
    from phonon_contribution import poly_fitting
    work_dir = r"C:\Users\dugue\OneDrive\spin-spin\1226_strain"
    atom_list = [1, 3, 421, 42, 362, 231]
    atom_name = ["$^{14}N$", "$^{13}C(1)$", "$^{13}C(2)$", "$^{13}C(3)$", "$^{13}C(4)$", "$^{13}C(5)$"]
    direction = np.array([1, 1, 1])
    direction = direction / np.linalg.norm(direction)
    fit_res_colle = {}
    for json_filename in os.listdir(work_dir):
        if json_filename == "xx+yy+zz.json" and "dontread" not in json_filename:
            print(json_filename[:-5])
            fit_res_colle[json_filename[:-5]] = {}
            data = json.load(open(f"{work_dir}/{json_filename}", "r"))
            for obj in data:
                obj["delta"] = float(obj["path"].split("/")[1])
            data.sort(key=lambda obj: obj["delta"])
            delta_list = np.array([obj['delta'] for obj in data])
            x = np.linspace(delta_list[0], delta_list[-1], 100)
            zero_index = np.where(delta_list == 0)[0][0]

            # energy = np.array([obj['energy'] for obj in data]) - data[zero_index]['energy']
            # hyperfine = {atom: np.array([obj['hyperfine_v'][str(atom)] for obj in data]) for atom in atom_list}
            hyperfine = {atom: np.array([
                np.linalg.norm(np.array(obj['hyperfine_tensor'][str(atom)]).dot(direction)) * np.sign(obj['fermi_contact'][str(atom)])
                for obj in data]) for atom in atom_list}

            fig = plt.figure()

            # A for 6 nuclei
            ax_list = [fig.add_subplot(panel) for panel in range(237)[231:]]
            for i in range(6):
                _coef, _error, _=poly_fitting(delta_list, hyperfine[atom_list[i]], 1)
                fit_res = {"fit": _coef.tolist(), "error": _error.tolist()}
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
            plt.subplots_adjust(hspace=0.8)
            plt.savefig(f"{work_dir}/strain_{'.'.join(json_filename.split('.')[:-1])}.png", dpi=300)
            plt.close("all")
    json.dump(fit_res_colle, open(f"{work_dir}/dontread_fit_res_colle3.json", "w"), indent=4)


if __name__ == "__main__":
    # work_dir = r"C:\Users\dugue\OneDrive\spin-spin\444_520_PS"
    # title_list = ["element_0_0", "element_0_1", "element_3_3", "element_3_4", "element_0_4", "element_0_3"]
    #
    # hyper_strain = np.array([[-0.5, -66, 46.9, 14.1, -27.6, -12.9],
    #                          [-0.5, -255, -10.1, 8.4, -7.7, 1.7],
    #                          [-0.5, -256, -12.2, 8.5, -3.2, 1.7],
    #                          [7.5, -211, -0.4, 2.3, 1.0, 17.1],
    #                          [7.4, 193, -2.0, -7.8, -15.5, -11.2],
    #                          [7.5, 191, -1.0, -7.8, -0.9, -11.2]])
    # hyper_strain *= 1e3
    # kBar2GPa = 0.1013
    # # --------------------------------------------------------------------------
    # # read_elastic_tensor(work_dir + "/elastic_tensor_nv_diamond_444",
    # #                    work_dir + "/elastic_tensor_nv_diamond_444.json")
    # # --------------------------------------------------------------------------
    # # fitting_elastic_tensor()
    # # --------------------------------------------------------------------------
    #
    # elastic_tensor, error = get_elastic_tensor(0.0036800710943974)
    # print(elastic_tensor, error)
    # elastic_coef, error_ = get_elastic_coef(0.0036800710943974)
    # print(elastic_coef, error_)
    # print(30 / 0.1013 / elastic_coef)
    # -------------------------------------------------------------------------------

    # stress = elastic_moduli_in_kBar.dot(strain)
    # elastic_moduli_in_kBar = np.array(
    #     [[1.01453699e+04, 1.26471400e+03, 1.26471400e+03, 3.93870000e+00, 4.91080000e+00, 3.93870000e+00],
    #      [1.26471400e+03, 1.01453699e+04, 1.26471400e+03, 3.93870000e+00, 3.93870000e+00, 4.91080000e+00],
    #      [1.26471400e+03, 1.26471400e+03, 1.01453699e+04, 4.91080000e+00, 3.93870000e+00, 3.93870000e+00],
    #      [3.93870000e+00, 3.93870000e+00, 4.91080000e+00, 5.39007660e+03, -9.52600000e-01, -9.52600000e-01],
    #      [4.91080000e+00, 3.93870000e+00, 3.93870000e+00, -9.52600000e-01, 5.39007660e+03, -9.52600000e-01],
    #      [3.93870000e+00, 4.91080000e+00, 3.93870000e+00, -9.52600000e-01, -9.52600000e-01, 5.39007660e+03]])

    # elastic_moduli_in_kBar = np.array([[10652, 1224, 1224, 0, 0, 0],
    #                                    [1224, 10652, 1224, 0, 0, 0],
    #                                    [1224, 1224, 10652, 0, 0, 0],
    #                                    [0, 0, 0, 5706, 0, 0],
    #                                    [0, 0, 0, 0, 5706, 0],
    #                                    [0, 0, 0, 0, 0, 5706]])
    # elastic_moduli_inv_in_kBar = np.linalg.inv(elastic_moduli_in_kBar)

    # A = hyper_strain.dot(strain)
    # hyper_strain = np.array(
    #     [[-8.41945292e-01, -6.15639270e+01, -7.88820088e+00, 5.42905315e+00, -8.04837226e+00, -1.16958796e+01],
    #      [-8.22700327e-01, -2.49565053e+02, -1.80406526e+01, 1.28679902e+01, -1.59003403e+00, 7.99076927e-01],
    #      [-8.19668240e-01, -2.49367173e+02, 4.94972021e+01, 1.28511792e+01, -2.56465936e+01, 8.07103006e-01],
    #      [7.92815871e+00, -2.18439561e+02, 4.78148383e-03, 2.66374640e+00, -1.65807981e+01, 1.65446945e+01],
    #      [7.91379048e+00, 1.89820080e+02, -1.96502419e+00, -7.32920774e+00, -1.43250740e+00, -1.10943914e+01],
    #      [7.95933675e+00, 1.91049751e+02, -9.29890947e-01, -7.38184456e+00, 1.88146858e+00, -1.10917362e+01]])
    # hyper_strain = np.array([[-0.52678178, -254.37095133, -12.3272319, 8.45663437, -7.68536719, -12.94496392],
    #                          [-0.4842665, -65.86223976, -10.0824711, 8.39433194, -3.16080846, 1.71628068],
    #                          [-0.49304466, -256.09476127, 47.00300289, 14.10387601, -27.59664386, 1.74140626],
    #                          [7.47924767, 190.7263894, -0.85132386, -7.72113814, -15.63628309, 17.08303804],
    #                          [7.42468153, -214.26643203, -1.99796778, -7.76139314, -0.8998618, -11.23450311],
    #                          [7.54172996, 191.12162192, -0.25820442, 2.34978923, 0.89230545, -11.23512471]])

    # for i in [True, False]:
    #     for j in range(6):
    #         stress_ellipsoid(normal=i, atom=j)
    # print(elastic_moduli_inv_in_kBar.dot(np.array([30, 30, 30, 0, 0, 0]).T / kBar2GPa))

    # -----------------------------------------------
    fit_dA_depslion()
