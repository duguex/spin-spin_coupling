# cython:language_level=3
import logging
import os
import numpy as np
import random
from copy import deepcopy
from collections import namedtuple
import json

np.set_printoptions(threshold=np.inf)


class Poscar:
    def __init__(self, poscar_path):
        assert type(poscar_path) == str and os.path.isfile(poscar_path), \
            "The parameter should be the path of a POSCAR-class file."
        self.read(poscar_path)

    def read(self, poscar_path):
        with open(poscar_path, "r") as poscar_file:
            # haven't read "\n"
            self.comment = poscar_file.readline().strip()
            self.scaling = float(poscar_file.readline().strip())
            self.lattice = poscar_file.readline() + poscar_file.readline() + \
                           poscar_file.readline()
            self.lattice = np.array(
                list(filter(None, self.lattice.strip().split(" "))), dtype='<f8').reshape(3, 3)

            self.omega = np.linalg.det(self.lattice) / 0.52918 ** 3
            # bohr_to_angstrom = 0.52918
            self.R1, self.R2, self.R3 = self.lattice / 0.52918
            self.element = [num
                            for num in poscar_file.readline().strip().split(" ") if num]
            self.number = [int(num)
                           for num in poscar_file.readline().strip().split(" ") if num]

            self.element = [num for num in poscar_file.readline().strip().split(" ") if num]

            self.number = np.array(poscar_file.readline().strip().split(" "), dtype="<i4")

            self.name = "".join([element + (str(number) if not number == 1 else "") for element, number in
                                 zip(self.element, self.number)])
            # read selective dynamics and type = Direct or Cartesian
            selective = poscar_file.readline().strip()
            if selective[0] in ["S", "s"]:
                self.selective = True
                self.type = poscar_file.readline().strip()
            else:
                self.selective = False
                self.type = selective

            position = []
            self.additional = []

            while True:
                line = poscar_file.readline().strip()
                if line:
                    line = [num for num in line.split(" ") if num]
                    position.append([float(num) for num in line[:3]])
                    self.additional.append(line[3:])
                else:
                    break

            self.position = np.array(position)

            self.chg_density_difference = []
            while True:
                line = poscar_file.readline()
                # EOF
                if not line or "0.00000000E+00" in line:
                    break
                else:
                    line = line.strip()
                    if line:
                        self.chg_density_difference = [num for num in line.split(" ") if num]
                        if len(self.chg_density_difference) == 3:
                            self.chg_density_difference = [int(num) for num in self.chg_density_difference]
                            break

            # read CHGCAR
            if len(self.chg_density_difference) == 3:
                self.chg = []
                while True:
                    line = poscar_file.readline()
                    if line:
                        line = line.strip()
                        if line:
                            self.chg += [float(num) for num in line.split(" ") if num]
                    else:
                        break

        self.reciprocal = np.linalg.inv(self.lattice)
        # bohr_to_angstrom = 0.52918
        self.G1, self.G2, self.G3 = self.reciprocal * 2 * np.pi * 0.52918

        self.label = []
        for element, element_number in zip(self.element, self.number):
            for number in range(element_number):
                # in case element is separated
                while (element, number) in self.label:
                    number += 1
                self.label.append((element, number))

        # reform chg
        if len(self.chg_density_difference) == 3:
            self.chg = [[num % self.chg_density_difference[0],
                         num // self.chg_density_difference[0] % self.chg_density_difference[1],
                         num // self.chg_density_difference[0] // self.chg_density_difference[1],
                         self.chg[num]] for num in range(len(self.chg))]
        else:
            self.chg_density_difference = []

        return None

    def write(self, poscar_path):
        poscar_string = f"{self.comment}\n{self.scaling}\n"
        poscar_string += self.lattice.__str__().replace("[", "").replace("]", "")
        poscar_string += "\n"
        poscar_string += " ".join(self.element)
        poscar_string += "\n"
        poscar_string += " ".join([str(num) for num in self.number])
        poscar_string += "\n"
        if self.selective:
            poscar_string += "Selective\n"
        poscar_string += f"{self.type}\n"
        poscar_string += np.append(self.position, self.additional, axis=1).__str__() \
            .replace("'\n  '", " ").replace("[", "").replace("]", "").replace("'", "")

        poscar_string += "\n\n"

        if self.chg_density_difference:
            poscar_string += " ".join([str(num) for num in self.chg_density_difference])
            poscar_string += "\n"
            poscar_string += " ".join([str(num) for num in list(zip(*self.chg))[-1]])
            poscar_string += "\n"

        with open(poscar_path, "w") as f:
            f.write(poscar_string)
        return None

    def d2c(self):
        aposcar = Poscar(self)
        if aposcar.type[0] in ["D", "d"]:
            aposcar.position = aposcar.position.dot(aposcar.lattice)
            aposcar.type = "C"
        return aposcar

    def c2d(self):
        aposcar = Poscar(self)
        if aposcar.type[0] in ["C", "c"]:
            aposcar.position = aposcar.position.dot(aposcar.reciprocal)
            for pos in aposcar.position:
                for num in range(3):
                    if pos[num] >= 1:
                        pos[num] -= 1
                    elif pos[num] < 0:
                        pos[num] += 1
            aposcar.type = "D"
        return aposcar

    def __add__(self, other):
        if isinstance(other, Poscar):
            aposcar = self.d2c()
            aposcar.position += other.position
            aposcar.lattice += other.lattice
            aposcar.reciprocal = np.linalg.inv(aposcar.lattice)
        else:
            aposcar = Poscar(self)
            # including vector and number
            aposcar.position += other

            # deal with chg only when object(other) is a vector
            if self.chg_density_difference:
                diff_other = [round(i * j) for i, j in zip(other, self.chg_density_difference)]
                for chg_den_diff in self.chg:
                    for i in range(3):
                        chg_den_diff[i] += diff_other[i]
                        if chg_den_diff[i] >= self.chg_density_difference[i]:
                            chg_den_diff[i] -= self.chg_density_difference[i]
                        elif chg_den_diff[i] < 0:
                            chg_den_diff[i] += self.chg_density_difference[i]

                self.chg.sort(key=lambda chg_den_diff: chg_den_diff[0] + (
                        chg_den_diff[1] + chg_den_diff[2] * self.chg_density_difference[1]) *
                                                       self.chg_density_difference[0])

            for pos in aposcar.position:
                for num in range(3):
                    if pos[num] >= 1:
                        pos[num] -= 1
                    elif pos[num] < 0:
                        pos[num] += 1

        return aposcar

    def __sub__(self, other):
        if isinstance(other, Poscar):
            other = other.c2d()
            aposcar = self.c2d()
            for pos_other, pos_apos in zip(other.position, aposcar.position):
                for num in range(3):
                    if pos_other[num] - pos_apos[num] > 0.5:
                        pos_other[num] -= 1
                    elif pos_other[num] - pos_apos[num] < -0.5:
                        pos_other[num] += 1
            other = other.d2c()
            aposcar = aposcar.d2c()
            aposcar.position -= other.position
            aposcar.lattice -= other.lattice
        else:
            aposcar = Poscar(self)
            aposcar.position = self.position - other
            for pos in aposcar.position:
                for num in range(3):
                    if pos[num] > 0.5:
                        pos[num] -= 1
                    elif pos[num] < -0.5:
                        pos[num] += 1
        return aposcar

    def __mul__(self, multi):
        aposcar = Poscar(self)
        aposcar.position *= multi
        if aposcar.type[0] in ["C", "c"]:
            aposcar.lattice *= multi

        return aposcar

    def __rmul__(self, multi):
        return self * multi

    def chg_density_add(self, other):
        aposcar = Poscar(self)
        for i in range(len(aposcar.chg)):
            aposcar.chg[i][-1] += other.chg[i][-1]
        return aposcar

    # convert tuple or index form of a point to a list / np.ndarray one
    def every2pos(self, site):
        if type(site) in [list, np.ndarray]:
            pass
        else:
            if type(site) == tuple:
                site = self.label.index(site)
            site = self.position[site]
        return site

    def distance(self, site):
        atom_tuple = namedtuple("atom_tuple", ["label", "distance", "position"])
        # 相对坐标!!
        # count from 0 not 1
        site = self.every2pos(site)
        distance_list = []
        for label, position in zip(self.label, (self - site).d2c().position):
            # 距离
            distance_list.append(atom_tuple(label=label, distance=np.linalg.norm(position), position=position))

        # sort by element
        distance_list.sort(key=lambda atom: atom.label)
        # sort by distance
        distance_list.sort(key=lambda atom: atom.distance)
        return distance_list

    def env(self, site, prec=0.1):
        env_tuple = namedtuple("env_tuple", ["distance", "counter", "specific_site"])
        env_list = []
        for atom in self.distance(site):
            distance = round(round(atom.distance / prec) * prec, 3)
            for env in env_list:
                if distance == env.distance:
                    if atom.label[0] in env.counter:
                        env.counter[atom.label[0]] += 1
                    else:
                        env.counter[atom.label[0]] = 1
                    env.specific_site.append(atom.label)
                    break
            else:
                env_list.append(env_tuple(distance=distance, counter={atom.label[0]: 1}, specific_site=[atom.label]))

        return env_list

    def same_site(self, env1, env2):
        for aenv1, aenv2 in list(zip(env1, env2)):
            if aenv1.distance == aenv2.distance and aenv1.counter == aenv2.counter:
                continue
            else:
                return False
        return True

    def sites(self, element, nei=3, prec=0.1):
        site_tuple = namedtuple("site_tuple", ["env", "specific_site"])
        site_list = []
        for label in self.label:
            if element == label[0]:
                env = self.env(label, prec)[:nei]
                for site in site_list:
                    if self.same_site(env, site.env):
                        site.specific_site.append(label)
                        break
                else:
                    site_list.append(site_tuple(env=env, specific_site=[label]))

        return site_list

    def distort_sphere(self, distort_list=[0.1], site=0):
        aposcar = Poscar(self)
        site = self.every2pos(site)
        for atom, distort in zip(self.distance(site)[1:], distort_list):
            aposcar.position[self.label.index(atom.label)] = atom.position.dot(self.reciprocal) * (
                    1 + distort / atom.distance) + site
        return aposcar

    def random_distort(self, displace=0.1, site=0, nei=3):
        aposcar = Poscar(self)
        aposcar.position = np.zeros((len(aposcar.position), 3))
        aposcar.type = "C"
        env = self.env(site, 0.1)
        for neighbours in env[:nei]:
            for label in neighbours.specific_site:
                vector = np.array([random.random() for _ in range(3)]) - .5
                vector /= np.linalg.norm(vector)
                aposcar.position[self.label.index(label)] = displace * random.random() * vector

        return self + aposcar.c2d()

    def atom(self, element):
        aposcar = Poscar(self)
        aposcar.element = [element]
        aposcar.number = [1]
        aposcar.position = np.array([[0.5, 0.5, 0.5]])
        return aposcar

    # configuration coordinate curves
    def cc(self, other, ini, fin, step):
        poscar_list = []
        for i in range(step):
            factor = round(ini + i * (fin - ini) / (step - 1), 2)
            poscar_list.append([factor, self + other * factor])
        return poscar_list

    # under construct
    def inter(self, prec=0.1, loop=1000):
        inter_tuple = namedtuple("inter_tuple", ["env", "counter", "poscar"])
        inter_list = []

        for i in range(loop):
            # 生成出生点
            current_site = np.array([random.random() for _ in range(3)])
            count = 1
            neighbours = self.distance(current_site)

            while True:
                # 移动远离最近邻
                new_site = current_site + 0.01 / (count / 100 if count > 100 else 1) * (
                        np.array([random.random() for _ in range(3)]) - .5)
                for i in range(3):
                    if new_site[i] < 0:
                        new_site[i] += 1
                    elif new_site[i] >= 1:
                        new_site[i] -= 1

                new_neighbours = self.distance(new_site)

                if new_neighbours[0].distance > neighbours[0].distance:
                    current_site = new_site
                    count = 1
                    neighbours = new_neighbours
                else:
                    count += 1

                if count > 500:
                    break

            current_env = self.env(current_site, prec)[:2]
            # 是否是重复的格位

            for inter_site in inter_list:
                if self.same_site(inter_site.env, current_env):
                    inter_site.counter[0] += 1
                    break
            else:
                aposcar = Poscar(self)
                aposcar.element.insert(0, "i")
                aposcar.number.insert(0, 1)
                aposcar.position = np.insert(aposcar.position, 0, current_site, axis=0)
                aposcar.additional.insert(0, [])
                print(current_site, current_env[0])
                inter_list.append(inter_tuple(env=current_env, counter=[1], poscar=aposcar))
        return inter_list

    def sub(self, origin_site, new_site):
        sub_tuple = namedtuple("sub_tuple", ["site", "poscar"])
        sub_list = []
        for site in self.sites(element=origin_site, nei=3):
            aposcar = Poscar(self)
            origin_element_index = aposcar.element.index(origin_site)
            if aposcar.number[origin_element_index] == 1:
                del aposcar.element[origin_element_index]
                del aposcar.number[origin_element_index]
            else:
                aposcar.number[origin_element_index] -= 1
            index = aposcar.label.index(site.specific_site[0])
            origin_position = aposcar.position[index]
            aposcar.position = np.delete(aposcar.position, index, 0)
            del aposcar.additional[index]
            if not new_site == "vac":
                aposcar.element.insert(0, new_site)
                aposcar.number.insert(0, 1)
                aposcar.position = np.insert(aposcar.position, 0, origin_position, axis=0)
                aposcar.additional.insert(0, [])
                aposcar = aposcar + (0.5 - origin_position)

            aposcar.label = []
            for element, element_number in zip(aposcar.element, aposcar.number):
                for number in range(element_number):
                    # in case element is separated
                    while (element, number) in aposcar.label:
                        number += 1
                    aposcar.label.append((element, number))
            sub_list.append(sub_tuple(site=site, poscar=aposcar))
        return sub_list

    # under constuct
    def slab(self, vac_length, multi=5):
        # z方向扩展2倍
        self.number = [multi * i for i in self.number]
        c_length = np.linalg.norm(self.lattice[-1])
        ratio = (c_length * multi + vac_length) / c_length
        for i in range(len(self.lattice[-1])):
            self.lattice[-1][i] *= ratio
        new_pos = []
        for i in range(len(self.position)):
            for k in range(multi):
                new_pos.append(
                    np.array([(self.position[i][j] + k) / ratio if j == 2 else self.position[i][j] for j in
                              range(3)]))
        self.position = new_pos
        return None

    def slab2(self, vac_length):
        self.number = [2 * i for i in self.number]
        c_length = np.linalg.norm(self.lattice[-1])
        ratio = (c_length * 2 + vac_length) / c_length
        for i in range(len(self.lattice[-1])):
            self.lattice[-1][i] *= ratio
        move_list = []
        for i in self.layer[-1]:
            if abs(i[2][0] - i[2][1]) < 0.01:
                move_list.append((i[0], i[1]))

        new_pos = []
        for i, j in zip(self.position, self.label):
            new_pos.append(np.array([i[j] / ratio if j == 2 else i[j] for j in range(3)]))
            if j in move_list:
                new_pos.append(np.array([1 + (i[j] - 1) / ratio if j == 2 else i[j] for j in range(3)]))
            else:
                new_pos.append(np.array([(i[j] + 1) / ratio if j == 2 else i[j] for j in range(3)]))
        self.position = new_pos
        return None

    def expand(self, tm):
        # transform_matrix

        self.lattice = np.dot(tm, self.lattice)
        # self.label=[(i,k+1) for i,j in zip(self.element,self.number) for k in range(j)]
        for i in range(len(self.number)):
            self.number[i] = 0
        new_positions = []
        for i, j in zip(self.label, self.position):
            for x in range(-3, 4):
                for y in range(-3, 4):
                    for z in range(-3, 4):
                        a_new_position = (j + np.array([x, y, z])).dot(np.linalg.inv(tm))
                        if max(a_new_position) < 1 and min(a_new_position) >= 0:
                            new_positions.append(a_new_position)
                            self.number[self.element.index(i[0])] += 1
        self.position = new_positions
        return None

    def ecp(self, center_atom, atom_number):
        # 直接输出到inp
        aposcar = self.c2d()
        aposcar = aposcar + (0.5 - aposcar.position[aposcar.label.index((center_atom, 1))])
        aposcar = aposcar.d2c()
        centered_env = aposcar.env((center_atom, 1), prec=0.1)
        actual_number = False
        sphere_position = []
        sphere_element = []
        sphere_number = []

        for radius in centered_env:
            for atom_label in centered_env[radius]:
                sphere_position.append(aposcar.position[aposcar.label.index(atom_label)])
                if not sphere_element or not atom_label[0] == sphere_element[-1]:
                    sphere_element.append(atom_label[0])
                    sphere_number.append(1)
                else:
                    sphere_number[-1] += 1

            if not actual_number and len(sphere_position) > atom_number:
                actual_number = len(sphere_position)
                actual_radius = radius
                break
        aposcar.position = np.array(sphere_position)
        aposcar.element = sphere_element
        aposcar.number = sphere_number

        return aposcar, actual_number, actual_radius


if __name__ == '__main__':
    p = Poscar("CONTCAR_Hf")
    print(p.position[:5])
    print(p.c2d().position[:5])
    print(p.d2c().position[:5])
    print(p.d2c().c2d().position[:5])
    # print(p.distance(p.label[0]))
    # print(p.distance([0, 0, 0]))
    # print(p.sites(site="Cl", nei=2))
    # print(p.sub("Cl","vac"))
    # print(p.inter("Cl",prec=0.2))

    # __sub__ test
    # a1 = Poscar("CONTCAR1")
    # a2 = Poscar("CONTCAR2")
    # print(a1.position[0])
    # print(a2.position[0])
    # print((a1 - a2).position[0])
