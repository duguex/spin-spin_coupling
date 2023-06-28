# -*- coding: utf-8 -*-
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MaxNLocator
from sympy import *
import os

from decompose_stress import rotate_stress_tensor
from phonon_contribution import poly_fitting
from uncertainties import unumpy

np.set_printoptions(suppress=True)


def plot_stress_ellipsoid(principal_stresses):
    # the principal stresses are the enigvalues of the stress tensor
    assert principal_stresses.shape == (3,)
    a, b, c = principal_stresses / 2

    # Define the angles for plotting
    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 100)

    # Define the coordinates of the ellipsoid
    x = a * np.outer(np.cos(phi), np.sin(theta))
    y = b * np.outer(np.sin(phi), np.sin(theta))
    z = c * np.outer(np.ones_like(phi), np.cos(theta))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='b')
    ax.set_xlabel('s1')
    ax.set_ylabel('s2')
    ax.set_zlabel('s3')
    ax.set_title('Stress Ellipsoid')
    plt.show()


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

    json.dump(elastic_tensor_dict, open(json_path, "w"))


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

    # C3v symmetry check
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
    tensor = unumpy.uarray(np.zeros((6, 6)), np.zeros((6, 6)))
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


def fitting_elastic_tensor(to_be_fitted, fitting_result):
    elastic_tensor_dict = json.load(open(to_be_fitted, "r"))
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
        ax.set_title(c3v_name_list[tensor_index])
    plt.subplots_adjust(hspace=0.8)
    plt.show()

    json.dump([coef_list, error_list], open(fitting_result, "w"))


def get_elastic_tensor(coef_with_error, scale):
    elements_with_error = coef_with_error.T[0] * scale + coef_with_error.T[1]
    return construct_tensor_c3v(elements_with_error)


def fit_A0_strain():
    fitting_result = {}
    for filename in metadata_list:
        data = json.load(open(metadata_dir + "/" + filename, "r"))
        fitting_result[filename[:-5]] = {}

        # scale
        for obj in data:
            obj["scale"] = float(obj["path"].split("/")[1])
        data.sort(key=lambda obj: obj["scale"])
        scale_list = np.array([obj["scale"] for obj in data])

        # the shape of A0 is (nuclei, scale, 3, 3)
        A0_tensor = []
        for atom in atom_list:
            for obj in data:
                A0_tensor.append(obj['hyperfine_tensor'][str(atom)])
        A0_tensor = np.array(A0_tensor).reshape((len(atom_list), len(scale_list), 3, 3))
        A0_tensor = A0_tensor.transpose((0, 2, 3, 1)).reshape((-1, len(scale_list)))

        coef_list = []
        error_list = []
        for A0 in A0_tensor:
            coef, error, _ = poly_fitting(scale_list, A0, 1)
            coef_list.append(coef)
            error_list.append(error)
        coef_list = np.array(coef_list).reshape((len(atom_list), 3, 3, 2))
        error_list = np.array(error_list).reshape((len(atom_list), 3, 3, 2))
        fitting_result[filename[:-5]] = [coef_list.tolist(), error_list.tolist()]
    json.dump(fitting_result, open(work_dir + "/A0_strain_fit.json", "w"))


def get_A0_coefficient():
    A0_coefficient = []
    for atom in range(6):
        for index in ["xx", "yy", "zz", "xy", "yz", "zx"]:
            coef_list, error_list = fitting_result[index]
            # shape of coef_list is (atom, 3, 3, 2)
            coef_list = np.array(coef_list[atom])
            coef0, coef1 = coef_list.transpose((2, 0, 1))
            x = np.linspace(-.01, .01, 100)
            A0_tensor_fit = np.einsum("ij,l->lij", coef0, x) + coef1
            A0_fit = np.linalg.norm(A0_tensor_fit.dot(direction), axis=1) * np.sign(np.sum(A0_tensor_fit, axis=(1, 2)))
            coef, error, _ = poly_fitting(x, A0_fit, 1)
            A0_coefficient.append(coef[0])
    # shape of A0_coefficient is (index, atom)
    A0_coefficient = np.array(A0_coefficient).reshape((6, 6)).T
    json.dump(A0_coefficient.tolist(), open(work_dir + "/A0_coefficient.json", "w"))


def plot_A0_strain_fitting(to_be_plotted):
    for filename in to_be_plotted:
        coef_list, error_list = fitting_result[filename[:-5]]
        data = json.load(open(metadata_dir + "/" + filename, "r"))

        # scale
        for obj in data:
            obj["scale"] = float(obj["path"].split("/")[1])
        data.sort(key=lambda obj: obj["scale"])
        scale_list = np.array([obj["scale"] for obj in data])
        x = np.linspace(scale_list[0], scale_list[-1], 100)

        # the shape of A0 is (nuclei, scale, 3, 3)
        A0_tensor = []
        for atom in atom_list:
            # scatter
            for obj in data:
                A0_tensor.append(obj['hyperfine_tensor'][str(atom)])
        A0_tensor = np.array(A0_tensor).reshape((len(atom_list), len(scale_list), 3, 3))
        A0_scatter = np.linalg.norm(A0_tensor.dot(direction), axis=2) * np.sign(np.sum(A0_tensor, axis=(2, 3)))
        # shape of coef_list is (nuclei, 3, 3, 2)
        coef0, coef1 = np.array(coef_list).transpose((3, 0, 1, 2))
        # shape of A0_tensor_fit is (scale, nuclei, 3, 3)
        A0_tensor_fit = np.einsum("ijk,l->ijkl", coef0, x).transpose((3, 0, 1, 2)) + coef1
        A0_tensor_fit = A0_tensor_fit.transpose((1, 0, 2, 3))
        A0_fit = np.linalg.norm(A0_tensor_fit.dot(direction), axis=2) * np.sign(np.sum(A0_tensor_fit, axis=(2, 3)))

        fig = plt.figure()

        # A0 for 6 nuclei
        ax_list = [fig.add_subplot(panel) for panel in range(237)[231:]]
        for i in range(6):
            ax_list[i].scatter(scale_list, A0_scatter[i])
            ax_list[i].plot(x, A0_fit[i], "r")
            ax_list[i].yaxis.set_major_locator(MaxNLocator(6))

        fig.tight_layout()
        plt.subplots_adjust(hspace=0.8)
        # plt.savefig(f"{work_dir}/strain_{'.'.join(json_filename.split('.')[:-1])}.png", dpi=300)
        # plt.close("all")
        plt.show()


def hyperfine_ellipsoid(stress_tensor, origin_direction):
    # theta from 0 to Pi and phi from 0 to 2Pi

    theta = np.linspace(0.01, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)

    # shape of direction is (theta, phi, 3)
    direction = np.array([np.outer(np.sin(theta), np.cos(phi)),
                          np.outer(np.sin(theta), np.sin(phi)),
                          np.outer(np.cos(theta), np.ones_like(phi))]).transpose((1, 2, 0)).reshape((-1, 3))
    # shape of stress is (theta * phi, 3, 3)
    stress = np.array([rotate_stress_tensor(stress_tensor, origin_direction, _direction) for _direction in direction])
    # shape of stress is (6, theta * phi)
    stress = np.array([[_stress[0, 0], _stress[1, 1], _stress[2, 2],
                        2 * _stress[0, 1], 2 * _stress[1, 2], 2 * _stress[2, 0]] for _stress in stress]).T
    # shape of strain is (6, 6).dot(6, theta * phi) = (6, theta * phi)
    strain = elastic_tensor_inv.dot(stress)
    # shape of deltaA0 is (atom, 6).dot(6, theta * phi) = (atom, theta * phi)
    deltaA0 = A0_coefficient.T.dot(strain).reshape((len(atom_list), theta.shape[0], phi.shape[0]))
    for dA in deltaA0:
        # (theta, phi), (theta, phi, 3)
        x, y, z = np.einsum("ij,ijk->ijk", dA, direction.reshape((theta.shape[0], phi.shape[0], 3))).reshape((-1, 3)).T
        x = x.reshape((theta.shape[0], phi.shape[0]))
        y = y.reshape((theta.shape[0], phi.shape[0]))
        z = z.reshape((theta.shape[0], phi.shape[0]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z)

        # Set the axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Surface plot')

        # Show the plot
        plt.show()


if __name__ == "__main__":
    metadata_dir = r"C:\Users\dugue\Desktop\metadata"
    work_dir = r"C:\Users\dugue\OneDrive\spin-spin_coupling\444_520_PS"
    c3v_name_list = ["element_0_0", "element_0_1", "element_3_3", "element_3_4", "element_0_4", "element_0_3"]

    # real stress ellipsoid
    # plot_stress_ellipsoid(np.array([1,2,3]))

    # read_elastic_tensor(metadata_dir + "/elastic_tensor_nv_diamond_444",
    #                     work_dir + "/elastic_tensor_nv_diamond_444.json")
    #
    # fitting_elastic_tensor(work_dir + "/elastic_tensor_nv_diamond_444.json",
    #                        work_dir + "/fitting_elastic_tensor_nv_diamond_444.json")

    coef_list, error_list = json.load(open(work_dir + "/fitting_elastic_tensor_nv_diamond_444.json", "r"))
    coef_with_error = unumpy.uarray(coef_list, error_list)
    # elastic_tensor_with_error = get_elastic_tensor(coef_with_error, 0.0036800710943974)
    # print(elastic_tensor_with_error)
    # -------------------------------------------------------------------------------
    atom_list = [1, 3, 421, 42, 362, 231]
    metadata_list = ["xx.json", "yy.json", "zz.json", "xy.json", "yz.json", "zx.json",
                     "xx+yy+zz.json", "yz+zx+xy.json", "xx+yz.json"]
    atom_name = ["$^{14}N$", "$^{13}C(1)$", "$^{13}C(2)$", "$^{13}C(3)$", "$^{13}C(4)$", "$^{13}C(5)$"]
    direction = np.array([1, 1, 1])
    direction = direction / np.linalg.norm(direction)

    fitting_result = json.load(open(work_dir + "/A0_strain_fit.json", "r"))
    # fit_A0_strain()
    # get_A0_coefficient()
    # plot_A0_strain_fitting(["xy.json"])
    # shape = (index, atom) index in order of ["xx", "yy", "zz", "xy", "yz", "zx"]
    A0_coefficient = json.load(open(work_dir + "/A0_coefficient.json", "r"))
    A0_coefficient = np.array(A0_coefficient)
    elastic_tensor = unumpy.nominal_values(get_elastic_tensor(coef_with_error, 0))
    elastic_tensor_inv = np.linalg.inv(elastic_tensor)

    # stress = elastic_tensor.dot(strain)
    # A0 = A0_coefficient.dot(strain)
    hyperfine_ellipsoid(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), [1, 0, 0])
