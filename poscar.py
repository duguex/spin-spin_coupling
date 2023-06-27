import os
import numpy as np
import random
from copy import deepcopy
import json


def split_and_filter(string):
    string = string.strip()
    return list(filter(None, string.split(" ")))


class Poscar:
    def __init__(self, poscar_path):
        assert type(poscar_path) == str and os.path.isfile(poscar_path), \
            "The parameter should be path of POSCAR-class file."

        '''
        Ca4 O4                   # comment
        1.0                      # scaling
        9.678532 0.0 0.0         # lattice vector 1
        0.0 9.678532 0.0         # lattice vector 2
        0.0 0.0 9.678532         # lattice vector 3
        Ca                       # element list
        1                        # element number list
        Direct# or Cartesian     # coordinate type
        0.5 0.5 0.5              # coordinate list
        ...
        '''

        with open(poscar_path, "r") as poscar:
            self.comment = poscar.readline().strip()

            scaling = float(poscar.readline().strip())
            lattice = poscar.readline() + poscar.readline() + poscar.readline()
            lattice = np.array(split_and_filter(lattice))
            self.lattice = lattice.reshape(3, 3).astype(float) * scaling
            self.reciprocal = np.linalg.inv(self.lattice)

            self.element = np.array(split_and_filter(poscar.readline()))
            self.number = np.array(split_and_filter(poscar.readline())).astype(int)
            self.name = "".join(
                [_element + ("" if _number == 1 else str(_number))
                 for _element, _number in zip(self.element, self.number)])

            if self.name not in self.comment:
                self.comment += " " + self.name

            # read selective dynamics and type = Direct or Cartesian
            selective = poscar.readline().strip()
            if selective[0] in ["S", "s"]:
                self.selective = True
                self.coordinate = poscar.readline().strip()
            else:
                self.selective = False
                self.coordinate = selective
            self.coordinate = self.coordinate[0].upper()
            assert self.coordinate in ["D", "C"]

            # read atomic position
            position = []
            addition = []

            while True:
                line = poscar.readline().strip()
                if line:
                    line = list(filter(None, line.split(" ")))
                    position.append(line[:3])
                    addition.append(" ".join(line[3:]))
                else:
                    break

            self.position = np.array(position).astype(float)
            self.addition = np.array(addition) if addition[0] else None

            # read charge density
            charge_density = ""
            while True:
                line = poscar.readline()
                # MD extra (from CONTCAR only)
                # The predictor-corrector coordinates are only provided to continue a molecular dynamic run
                if line and "0.00000000E+00" not in line:
                    charge_density += line
                else:
                    break

            if charge_density:
                charge_density = split_and_filter(charge_density)
                _shape = np.array(charge_density[:3]).astype(int)
                self.charge_density = np.array(charge_density[3:]).astype(float).reshape(_shape)
            else:
                self.charge_density = None

    def dump(self, json_path):
        with open(json_path, "w") as json_file:
            json.dump(self.__dict__, json_file)

    def load(self, json_path):
        with open(json_path, "r") as json_file:
            self.__dict__ = json.load(json_file)

    def write(self, poscar_path):
        poscar_string = self.comment + "\n1.\n"
        poscar_string += self.lattice.__str__().replace("[", "").replace("]", "") + "\n"
        poscar_string += " ".join(self.element) + "\n"
        poscar_string += " ".join(np.array(self.number).astype("str")) + "\n"

        if self.selective:
            poscar_string += "S\n"
        poscar_string += f"{self.coordinate}\n"

        # the shape of self.position is (natom, 3), and self.addition is (natom,)
        for atom_str in np.concatenate((self.position, self.addition[:, np.newaxis]), axis=1):
            poscar_string += " ".join(atom_str) + "\n"
        poscar_string += "\n"

        if self.charge_density is not None:
            _shape = self.charge_density.shape.astype("str")
            poscar_string += " ".join(_shape) + "\n"

            # print charge density, 10 numbers per line
            number_per_line = 10
            # the totol number of charge density is not a multiple of number_per_line
            total_number = np.product(self.charge_density.shape)
            residue_number = total_number % number_per_line
            residue = self.charge_density[-residue_number:]
            reshaped_charge_density = self.charge_density[:-residue_number].reshape(-1, number_per_line)
            for chg_str in reshaped_charge_density:
                poscar_string += " ".join([str(a_float) for a_float in chg_str]) + "\n"
            poscar_string += " ".join([str(a_float) for a_float in residue]) + "\n"

        with open(poscar_path, "w") as f:
            f.write(poscar_string)

    def copy(self):
        return deepcopy(self)

    def d2c(self):
        if self.coordinate == "D":
            self.table["position"] = self.table["position"].dot(self.lattice)
            self.coordinate = "C"

    def c2d(self):
        if self.coordinate == "C":
            self.table["position"] = self.table["position"].dot(self.reciprocal)
            self.coordinate = "D"

    # self.coordinate = "D" required
    def standardize(self, diff=False):
        if self.coordinate == "C":
            self.c2d()
            d2c_at_last = True
        else:
            d2c_at_last = False
        position = self.table["position"].copy()
        position = np.where(position >= (0.5 if diff else 1), position - 1, position)
        self.table["position"] = np.where(position < (-0.5 if diff else 0), position + 1, position)
        if d2c_at_last:
            self.d2c()

    def __add__(self, a_Poscar_or_an_array):
        '''
        :param a_Poscar_or_an_array:
        :return:

        case 1: a_Poscar_or_an_array is Poscar, self and a_Poscar_or_an_array have chg
        ignore position and sum chg
        case 2: a_Poscar_or_an_array is Poscar, self and a_Poscar_or_an_array don't both have chg
        sum position
        case 3: a_Poscar_or_an_array is array, self has chg
        sum position and move chg
        case 3: a_Poscar_or_an_array is array, self hasn't chg
        sum position
        '''
        a_Poscar = self.copy()

        if isinstance(a_Poscar_or_an_array, (np.ndarray, list, tuple)):
            a_Poscar_or_an_array = np.array(a_Poscar_or_an_array)
            assert a_Poscar_or_an_array.shape == (3,), \
                f"the shape of the array should be (3,), f{a_Poscar_or_an_array.shape} is given."

            if isinstance(a_Poscar.charge_density, np.ndarray):
                # case 3
                a_Poscar.charge_density = np.roll(a_Poscar.charge_density,
                                                  np.multiply(a_Poscar.charge_density.shape,
                                                              a_Poscar_or_an_array).astype(int), axis=(2, 1, 0))
            else:
                # case 4
                a_Poscar.table["position"] += a_Poscar_or_an_array
                a_Poscar.standardize()


        else:
            assert isinstance(a_Poscar_or_an_array, Poscar), \
                "the parameter should be a Poscar-class or iterable object."

            if isinstance(a_Poscar.charge_density, np.ndarray) and \
                    isinstance(a_Poscar_or_an_array.charge_density, np.ndarray):
                # case 1
                assert np.all(a_Poscar.charge_density.shape == a_Poscar_or_an_array.charge_density.shape), \
                    f"the shapes of the two charge densities {a_Poscar.charge_density.shape} and {a_Poscar_or_an_array.charge_density.shape} don't matched."
                a_Poscar.charge_density += a_Poscar_or_an_array.charge_density
            else:
                # case 2
                assert a_Poscar.coordinate == a_Poscar_or_an_array.coordinate, "the coordinate of poscars are not match."
                a_Poscar.table["position"] += a_Poscar_or_an_array.table["position"]
                a_Poscar.standardize()

        return a_Poscar

    def __sub__(self, a_Poscar):
        assert isinstance(
            a_Poscar, Poscar), "the parameter should be a Poscar-class object"

        b_Poscar = self.copy()
        assert a_Poscar.coordinate == b_Poscar.coordinate, "the coordinate of poscars are not match."
        b_Poscar.table["position"] -= a_Poscar.table["position"]
        b_Poscar.standardize(diff=True)

        return b_Poscar

    # only for diff
    def __mul__(self, a_number):
        assert isinstance(a_number, (int, float)
                          ), "the parameter should be a real number"
        a_Poscar = self.copy()
        a_Poscar.table["position"] *= a_number

        return a_Poscar

    def __rmul__(self, a_number):
        return self * a_number

    # TODO: need distance
    def distance(self, a_position):
        if isinstance(a_position, (list, tuple, np.ndarray)):
            a_position = np.array(a_position)
            assert a_position.shape == (3,)
        else:
            assert isinstance(a_position, int), "parameter should be atom index or a coordinate"
            assert self.table.shape[0] > a_position >= 0, \
                f"atom index {a_position} out of range [0,{self.table.shape[0]})"
            a_position = self.table[a_position]["position"]

        a_position = self.every2pos(a_position)
        distance_list = []
        for label, position in zip(self.label, (self - a_position).d2c().position):
            # 距离
            distance_list.append(atom_tuple(
                label=label, distance=np.linalg.norm(position), position=position))

        # sort by element
        distance_list.sort(key=lambda atom: atom.label)
        # sort by distance
        distance_list.sort(key=lambda atom: atom.distance_from_point_to_line)
        return distance_list

    # TODO: need tags
    def env(self, site, prec=0.1):
        env_tuple = namedtuple(
            "env_tuple", ["distance", "counter", "specific_site"])
        env_list = []
        for atom in self.distance(site):
            distance = round(round(atom.distance_from_point_to_line / prec) * prec, 3)
            for env in env_list:
                if distance == env.distance:
                    if atom.label[0] in env.counter:
                        env.counter[atom.label[0]] += 1
                    else:
                        env.counter[atom.label[0]] = 1
                    env.specific_site.append(atom.label)
                    break
            else:
                env_list.append(env_tuple(distance=distance, counter={
                    atom.label[0]: 1}, specific_site=[atom.label]))

        return env_list

    # TODO: need cluster
    def same_site(self, env1, env2):
        for aenv1, aenv2 in list(zip(env1, env2)):
            if aenv1.distance_from_point_to_line == aenv2.distance_from_point_to_line and aenv1.counter == aenv2.counter:
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
                    1 + distort / atom.distance_from_point_to_line) + site
        return aposcar

    def random_distort(self, displace=0.1, site=0, nei=3):
        aposcar = Poscar(self)
        aposcar.position = np.zeros((len(aposcar.position), 3))
        aposcar.coordinate = "C"
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

                if new_neighbours[0].distance_from_point_to_line > neighbours[0].distance_from_point_to_line:
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
                aposcar.tag.insert(0, [])
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
            del aposcar.tag[index]
            if not new_site == "vac":
                aposcar.element.insert(0, new_site)
                aposcar.number.insert(0, 1)
                aposcar.position = np.insert(aposcar.position, 0, origin_position, axis=0)
                aposcar.tag.insert(0, [])
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
                new_pos.append(np.array([(self.position[i][j] + k) / ratio if j == 2 else self.position[i][j] for j in
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
            new_pos.append(
                np.array([i[j] / ratio if j == 2 else i[j] for j in range(3)]))
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
        aposcar = aposcar + \
                  (0.5 - aposcar.position[aposcar.label.index((center_atom, 1))])
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
    # test read, write, dump and load
    p = Poscar("../vasp/POSCAR678")
    print(p.__dict__)
    # p.distance([0.5, 0.5, 0.5])

    # print((p + [0.4, 0, 0]).__dict__)

    # print(p.position[:5])
    # print(p.c2d().position[:5])
    # print(p.d2c().position[:5])
    # print(p.d2c().c2d().position[:5])
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
