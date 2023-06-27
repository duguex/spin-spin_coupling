import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import poscar
import scc_lib


def read_modes_and_freqs(mode_path):
    freq = []
    mode = []

    try:
        band = open(mode_path, 'r')
    except OSError:
        print("Could not open band.yaml")
        sys.exit()

    for line in band:
        if "  band:" in line:
            break

    while True:
        if band.readline() == "\n":
            break
        else:
            freq.append(float(band.readline().strip().split()[1]) * 4.135669057 / 1e3)  # THz to eV
            band.readline()
            mode.append([])
            for a in range(len(ground.position)):
                band.readline()
                mode[-1].append(
                    np.array([float(band.readline().replace(",", "").strip().split()[2]) for j in range(3)]))

    band.close()

    return freq, mode


def job_sub():
    # run jobs
    for i in range(len(mode))[3:]:
        pos = ground.copy()
        pos.position = mode[i]
        for j in [-1, 1]:
            if f"d{i}_{j}" not in os.listdir():
                (ground + pos * (j / 10)).write(f"{i}_{j}")
                scc_lib.linux_command(f"mkdir d{i}_{j}; mv {i}_{j} d{i}_{j}/POSCAR; cp INCAR POTCAR KPOINTS d{i}_{j};")
                scc_lib.linux_command(f"cd d{i}_{j}; 0run -q {random.choice(['smallopa', 'smallib'])};cd ..;")
    # i=9999
    # pos1=ground.copy()
    # pos2 = ground.copy()
    # pos1.position = type[5]
    # pos2.position=type[10]
    # for j in range(-5,6):
    #     (ground + (pos1+pos2) * (j / 50)).write(f"{i}_{j}")
    #     scc_lib.linux_command(f"mkdir d{i}_{j}; mv {i}_{j} d{i}_{j}/POSCAR; cp INCAR POTCAR KPOINTS d{i}_{j};")
    #     scc_lib.linux_command(f"cd d{i}_{j}; 0run -q {random.choice(['smallopa','ckduan','smallib'])};cd ..;")


def read_job(freq, para):
    # read results
    Z_A2_per_pho = []
    keyword_dict = {
        0: f"grep 'Fermi contact (isotropic) hyperfine coupling parameter (MHz)' OUTCAR -A{para[2] + 3} | tail -n1",
        1: f"grep 'Dipolar hyperfine coupling parameters (MHz)' OUTCAR -A{para[2] + 3} | tail -n1",
        2: f"grep 'Total hyperfine coupling parameters after diagonalization (MHz)' OUTCAR -A{para[2] + 4} | tail -n1"}

    for i in range(len(mode))[3:]:
        A_tot = []
        energy = []
        for j in [-1, 0, 1]:
            if j == 0:
                os.chdir("d3_0")
            else:
                os.chdir(f"d{i}_{j}")
            # 第一/5个原子是14N， 第二/6个是13C
            A_tot.append(
                float(
                    [k for k in scc_lib.linux_command(keyword_dict[para[0]])[0].strip().split(" ") if k][para[1]]
                )
            )
            energy.append(
                float(
                    [k for k in scc_lib.linux_command("tail OSZICAR -n1")[0].strip().split(" ") if k][4]
                )
            )
            os.chdir("..")
        mid = len(energy) // 2
        temp = []
        nph = []
        error = []
        A2 = []
        A2_per_pho = []
        A1 = []
        for j in range(mid + 1)[1:]:
            # classical
            # temp.append((energy[mid+j]-energy[mid])/8.6173e-05)

            # if (energy[mid + j] - energy[mid])<1e-5:
            #     temp.append(0)
            # else:
            #     temp.append(freq[i]/(8.6173e-05*np.log(freq[i]/(energy[mid + j] - energy[mid])+1)))
            # nph.append((energy[mid + j] - energy[mid])/freq[i])
            # #temp_error.append((energy[j+mid]-energy[mid-j])/0.025852*300)
            # error.append(energy[j+mid]-energy[mid-j])
            # A1.append((A_tot[mid + j] - A_tot[mid - j]) / 2)
            A2.append((A_tot[mid + j] + A_tot[mid - j]) / 2 - A_tot[mid])
            deltaE = energy[mid + j] - energy[mid]
            deltaE = deltaE if deltaE > 1e-5 else 1e-5
            A2_per_pho.append(A2[-1] / deltaE * freq[i])

        Z_A2_per_pho.append(A2_per_pho)

        # temp=np.array(temp)
        # nph=np.array(nph)

        # print(f"type{i}",end=" ")

        # 一次项随振动幅值变化, 验证一次项
        # x = np.linspace(0, 9, 10)
        # a1 = np.polyfit(x, A1, 1)
        # print(*a1,end=" ")
        # plt.figure()
        # plt.scatter(x,A1)
        # plt.plot(x, np.poly1d(a1)(x),"--")
        # plt.xlabel('Configuration')
        # plt.ylabel('A1 (MHz)')
        # plt.title("there should be a linear relation")
        # plt.savefig(f'type{i}_a1c')

        # new
        # t = np.linspace(0, 500, 501)
        # for app in A2_per_pho:
        #     a2=
        # plt.figure()
        # plt.scatter(x, A2_per_pho)
        # plt.savefig(f'type{i}_app')

        # # 声子数
        # plt.figure()
        # plt.scatter(nph,A2)
        # plt.xlabel('number of phonon')
        # plt.ylabel('A2 (MHz)')
        # plt.title("there should be a linear relation")
        # plt.savefig(f'type{i}_a2n')
        #
        # # 二次项随振动幅值变化, 验证二次项
        # a2 = np.polyfit(x, A2, 2)
        # print(*a2,end=" ")
        # plt.figure()
        # plt.scatter(x,A2)
        # plt.plot(x, np.poly1d(a2)(x), "--")
        # plt.xlabel('Configuration')
        # plt.ylabel('A2 (MHz)')
        # plt.title("there should be a quadratic relation")
        # plt.savefig(f'type{i}_a2c')
        #
        # # 温度随振动幅值变化
        # t = np.polyfit(x, temp, 2)
        # print(*t,end=" ")
        # plt.figure()
        # plt.scatter(x,temp)
        # plt.plot(x, np.poly1d(t)(x), "--")
        # plt.xlabel('Configuration')
        # plt.ylabel('Temperature (K)')
        # plt.title("there should be a quadratic relation")
        # plt.savefig(f'type{i}_tc')
        #
        # # 二次项随温度变化
        # a2t = np.polyfit(temp[-2:],A2[-2:], 1)
        # print(*a2t)
        # plt.figure()
        # plt.plot(temp, np.poly1d(a2t)(temp), "--")
        # plt.scatter(temp,A2)
        # plt.xlabel('Temperature (K)')
        # plt.ylabel('A2 (MHz)')
        # plt.title("there should be a linear relation")
        # plt.savefig(f'type{i}_a2t')
        #
        # # 非谐部分随温度变化
        # plt.figure()
        # plt.scatter(temp,error)
        # plt.xlabel('Temperature (K)')
        # plt.ylabel('Anharmonic (eV)')
        # plt.savefig(f'type{i}_anharmonic')

        # plt.close("all")

    Z_A2_per_pho = list(zip(*Z_A2_per_pho))
    t = np.linspace(0, 500, 501)
    for A2_per_pho in range(len(Z_A2_per_pho)):

        atot = np.zeros(len(t))
        atot_per_t = np.zeros(len(t))
        for f, a2 in zip(freq[3:], Z_A2_per_pho[A2_per_pho]):
            print(f, a2)
            npf = 1 / (np.exp(f / (8.6173e-05 * t)) - 1)
            atot += (npf + 0.5) * a2
            atot_per_t += (npf + 1) * npf * a2 * f / 8.6173e-05 / np.power(t, 2)

        plt.figure()
        plt.scatter(t, atot)
        plt.xlabel('Temperature (K)')
        plt.ylabel('A2 (MHz)')
        plt.savefig('a2t_' + "_".join([str(i) for i in para]))

        plt.figure()
        plt.scatter(t, atot_per_t)
        plt.xlabel('Temperature (K)')
        plt.ylabel("A2\' (MHz/K)")
        plt.savefig('a2\'t_' + "_".join([str(i) for i in para]))

        # plt.figure()
        # plt.scatter(t[:-1],np.array([atot[i+1]-atot[i] for i in range(len(atot)-1)]))
        # plt.xlabel('Temperature (K)')
        # plt.ylabel("A2\' (MHz/K)")
        # plt.savefig(f'2a2t2t_{A2_per_pho}')

        plt.close("all")


ground = poscar.Poscar()
ground.read("pg.vasp")
freq, mode = read_modes_and_freqs("2band.yaml")

dict12 = {0: [1, 2, 3, 4, 5], 1: [1, 2, 3, 4, 5, 6], 2: [1, 2, 3]}
for i in dict12:
    for j in dict12[i]:
        for k in [1, 63, 89, 137]:
            para = [i, j, k]
            read_job(freq, para)
