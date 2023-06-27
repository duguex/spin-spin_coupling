import os
import random

from poscar import Poscar
import scc_lib


# grd = poscar.poscar("POSCAR")
# strain_dict = {"xx": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((3, 3)),
#                "yy": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape((3, 3)),
#                "zz": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape((3, 3)),
#                "yz": np.array([0, 0, 0,
#                                0, 0, 0.5,
#                                0, 0.5, 0]).reshape((3, 3)),
#                "zx": np.array([0, 0, 0.5,
#                                0, 0, 0,
#                                0.5, 0, 0]).reshape((3, 3)),
#                "xy": np.array([0, 0.5, 0,
#                                0.5, 0, 0,
#                                0, 0, 0]).reshape((3, 3))}

# def strain_poscar_sub(pos, strain_list, delta=np.linspace(-0.01, 0.01, 21)):
#     # strain_list is a 6, shape list
#     tag = "+".join([("" if i == 1 else str(i)) + j for i, j in zip(strain_list, strain_dict) if i]).replace("+-", "-")
#     strain = sum([i * j for i, j in zip(strain_list, strain_dict.values()) if i])
#     if not os.path.isdir(f"{tag}"):
#         scc_lib.linux_command(f"mkdir {tag}")
#     for d in delta:
#         if not os.path.isdir(f"{tag}/{d}"):
#             poscar_copy = poscar.poscar(pos)
#             poscar_copy.lattice = poscar_copy.lattice.dot(np.eye(3, 3) + strain * d)
#             poscar_copy.write("poscar")
#             scc_lib.linux_command(
#                 f"mkdir {tag}/{d}; "
#                 f"mv poscar {tag}/{d}/POSCAR; "
#                 f"cp INCAR POTCAR KPOINTS WAVECAR {tag}/{d};"
#                 f"cd {tag}/{d}; "
#                 f"0run -q {random.choice(['smallopa', 'ckduan'])};"
#                 f"cd ../..;")
#
#
# #
# for strain_list in [[1, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 1],
#                     [1, 1, 1, 0, 0, 0],
#                     [0, 0, 0, 1, 1, 1],
#                     [1, 0, 0, 1, 0, 0]]:
#     strain_poscar_sub(grd, strain_list)


def expand_sub():
    # run jobs, should run read_yaml first
    pos = poscar.Poscar("POSCAR")
    pos.c2d()
    for i in range(10):
        temp_pos=poscar.Poscar(pos)
        # 5/1000
        factor=i/2000+1
        temp_pos.lattice *= factor
        if f"z{factor}" not in os.listdir():
            temp_pos.write(f"z{factor}")
            scc_lib.linux_command(
                f"mkdir s{factor}; "
                f"mv z{factor} s{factor}/POSCAR; "
                f"cp INCAR POTCAR KPOINTS WAVECAR s{factor};"
                f"cd s{factor}; "
                f"0run -q {random.choice(['smallopa'])};"
                f"cd ..;")
    return None
expand_sub()
#
#
# def random_sub(displacement, label):
#     # run jobs, should run read_yaml first
#     if f"r{displacement}_{label}" not in os.listdir():
#         scc_lib.linux_command(f"mkdir r{displacement}_{label}; ")
#         ground.random_distort(displacement).write(f"r{displacement}_{label}/POSCAR")
#         scc_lib.linux_command(f"cp INCAR POTCAR KPOINTS WAVECAR r{displacement}_{label};"
#                               f"cd r{displacement}_{label}; "
#                               f"0run -q {random.choice(['smallopa'])};"
#                               f"cd ..;")
#     return None
#
#     elif pat == 4:
#         for i in range(5):
#             random_sub(0.1, i)
#             random_sub(0.05, i)
#             random_sub(0.01, i)



# lattice_list = [round(10.7+i*0.002,3) for i in range(25)]
# lattice_list = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]

# lattice_list = [70, 70.5, 71, 71.5, 72, 72.5, 73, 73.5, 74, 74.5, 75, 75.5, 76, 76.5, 77, 77.5, 78]
# for i in lattice_list:
#     expand_sub(i)

#     ground = poscar.poscar(r"D:\OneDrive\10.20\POSCAR").d2c()
#     if pat == 8:
#         # 分解随机位移测试
#
#         ground_c2d = ground.c2d()
#         energy0 = get_energy("d0_0")
#         # omega 就是声子能量
#         # print(sum(freq) / 2)
#
#         for j in [0.01, 0.05, 0.1]:
#             for i in range(5):
#                 path = f"r{j}_{i}"
#                 rp = poscar.poscar()
#                 rp.read(path + "/CONTCAR")
#                 distort = (rp - ground_c2d)
#                 distort.d2c()
#                 energy = 0
#                 # 包含了虚频模
#                 # for nmode in range(len(mode)):
#                 # 未包含虚频模
#                 for nmode in range(len(mode))[3:]:
#                     proj_on_specific_mode = sum(
#                         [distort.position[natom].dot(mode[nmode][natom]) * np.sqrt(mass[natom]) for natom in
#                          range(len(mass))])
#                     # print(nmode, proj_on_specific_mode ** 2)
#                     energy += (proj_on_specific_mode * freq[nmode]) ** 2
#                 energy *= C / 2
#                 print(energy, get_energy(path) - energy0)
