import json
import numpy as np
import os

'''
band.conf

BAND = 0 0 0 0 0 0
BAND_POINTS = 1
DIM = 1 1 1
EIGENVECTORS = .TRUE.
FORCE_CONSTANTS = READ
# PRIMITIVE_AXES = Auto

phonopy --fc vasprun.xml
echo -e 'BAND = 0 0 0 0 0 0\nBAND_POINTS = 1\nDIM = 1 1 1\nEIGENVECTORS = .TRUE.\nFORCE_CONSTANTS = READ' > band.conf
phonopy band.conf

python vasp2xsf.py -i OUTCAR -p POSCAR -m 0
'''

THz2eV = 4.135669057e-3


def read_phonon_from_yaml(yaml_path, json_path):
    with open(yaml_path, "r") as f:
        mass = []
        while True:
            line = f.readline()
            if "mass:" in line:
                mass.append(float(line.strip().split()[-1]))
            elif line == "\n":
                break

        number_of_atom = len(mass)
        print(f"number of atoms: {number_of_atom}")
        while True:
            line = f.readline()
            if "band:" in line:
                break

        frequency = []
        motion = []

        while True:
            line = f.readline()
            if line == "\n":
                break
            else:
                frequency.append(float(f.readline().strip().split()[1]) * THz2eV)
                f.readline()
                motion.append([])
                for atom in range(number_of_atom):
                    f.readline()
                    motion[-1].append(
                        [float(f.readline().strip().split()[2][:-1]) for _ in range(3)]
                    )

    print(f"number of phonon modes: {len(frequency)}")
    # norm check
    # for mode_index in range(len(frequency)):
    #     # For metrics Frobenius norm is not equal to 2-norm
    #     motion_norm = np.linalg.norm(np.array(motion)[mode_index, ...])
    #     assert np.isclose(motion_norm, 1), f"the motion of mode {mode_index} may not be normed, which is {motion_norm}."

    # 非质量加权的振动. 只有比例有意义, 值没有意义
    # modewom = [[vib_per_mode_per_atom / np.sqrt(mass_per_atom) for mass_per_atom, vib_per_mode_per_atom in
    #             zip(mass, vib_per_mode)] for vib_per_mode in factor]

    json.dump([mass, frequency, motion], open(json_path, "w"))
    return None


def read_phonon_from_outcar(outcar_path, json_path):
    """
        number of ions:
        number of dos      NEDOS =    301   number of ions     NIONS =    216

        Eigenvectors and eigenvalues of the dynamical matrix
        ----------------------------------------------------

        1 f  =   39.961084 THz   251.082895 2PiTHz 1332.958235 cm-1   165.265818 meV
        X         Y         Z           dx          dy          dz
        0.889301  0.889561  0.889102     0.069023   -0.000836   -0.001398

        648 f/i=    0.085721 THz     0.538598 2PiTHz    2.859329 cm-1     0.354512 meV
        X         Y         Z           dx          dy          dz
        0.889301  0.889561  0.889102    -0.024965   -0.052717   -0.035156

    """

    with open(outcar_path, "r") as f:

        """
        get the number of ions
        
        number of dos      NEDOS =    301   number of ions     NIONS =    216
        """

        while True:
            line = f.readline()
            if "number of ions" in line:
                break

        number_of_ions = int(list(filter(None, line.strip().split()))[-1])
        number_of_modes = 3 * number_of_ions

        """
            get the number and mass of ions for each type
            
            ions per type =               1 510
        """

        while True:
            line = f.readline()
            if "ions per type =" in line:
                break
        ions_per_type = list(map(int, list(filter(None, line.strip().split()))[4:]))
        assert sum(ions_per_type) == number_of_ions, \
            "the total count of ions (ions per type) should equal to their total number (number of ions)."

        """
            get the number and mass of ions for each type

            Mass of Ions in am
            POMASS =  14.00 12.01
        """

        while True:
            line = f.readline()
            if "Mass of Ions in am" in line:
                break
        mass_per_type = list(map(float, list(filter(None, f.readline().strip().split()))[2:]))
        assert len(mass_per_type) == len(ions_per_type), \
            "the type of ions (ions per type and Mass of Ions) should be the same."
        mass = []
        for m, n in zip(mass_per_type, ions_per_type):
            mass += [m] * n

        # skip to phonon part

        phonon_flag = r"Eigenvectors and eigenvalues of the dynamical matrix"
        motion_flag = r"X         Y         Z           dx          dy          dz"

        while True:
            if phonon_flag in f.readline():
                break

        while True:
            line = f.readline()
            if "THz" in line:
                break

        frequency = []
        motion = []

        while True:
            """
            we are at here now --> 1 f  =   39.961084 THz   251.082895 2PiTHz 1332.958235 cm-1   165.265818 meV
                                             X         Y         Z           dx          dy          dz
                                      0.889301  0.889561  0.889102     0.069023   -0.000836   -0.001398
            """

            if "THz" in line:
                current_frequency = float(list(filter(None, line.split()))[-2])
                if "f/i" in line:
                    current_frequency = -current_frequency
                frequency.append(current_frequency)
                motion.append([])

            elif motion_flag in line or not line:
                pass
            elif "--" in line or "ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)" in line:
                break
            else:
                motion[-1].append(list(map(float, list(filter(None, line.split()))[-3:])))

            line = f.readline().strip()

    assert len(frequency) == number_of_modes, "wrong shape of frequencies"
    assert np.array(motion).shape == (number_of_modes, number_of_ions, 3), "wrong shape of motions"
    frequency = list(map(lambda x: x / 1000, frequency[::-1]))
    motion = motion[::-1]
    json.dump([mass, frequency, motion], open(json_path, "w"))


def match_lines(fp, content):
    fp.seek(0, 0)
    current_index = -1
    last_index = -1
    index_list = []
    while True:
        line = fp.readline()
        if line:
            last_index = current_index
            current_index = fp.tell()
            if content in line:
                index_list.append(last_index)
        else:
            break

    return index_list


def read_hyperfine_from_outcar(outcar_path, factor):
    """

        atom_list picks representive atoms

    """

    with open(outcar_path, "r") as f:
        """

             3112:  VOLUME and BASIS-vectors are now :
             6187:  VOLUME and BASIS-vectors are now :
             9272:  VOLUME and BASIS-vectors are now :
            12349:  VOLUME and BASIS-vectors are now :

        5341
         VOLUME and BASIS-vectors are now :
         -----------------------------------------------------------------------------
          energy-cutoff  :      520.00
          volume of cell :     2880.93
              direct lattice vectors                 reciprocal lattice vectors
            14.229100900  0.002683912  0.002683912     0.070278514 -0.000013254 -0.000013254
             0.002683912 14.229100900  0.002683912    -0.000013254  0.070278514 -0.000013254
             0.002683912  0.002683912 14.229100900    -0.000013254 -0.000013254  0.070278514

          length of vectors
            14.229101406 14.229101406 14.229101406     0.070278517  0.070278517  0.070278517
        """

        f.seek(match_lines(f, "length of vectors")[-1], 0)
        f.readline()
        length_of_vectors = list(map(float, list(filter(None, f.readline().strip().split()))[:3]))

        """
        6393
          FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
          ---------------------------------------------------
          free  energy   TOTEN  =     -4899.66895334 eV

          energy  without entropy=    -4899.66895334  energy(sigma->0) =    -4899.66895334



        --------------------------------------------------------------------------------------------------------
        """
        f.seek(match_lines(f, "free  energy   TOTEN")[-1], 0)
        energy = float(list(filter(None, f.readline().strip().split()))[-2])

        """
        10514

                   NMR quadrupolar parameters

             Cq : quadrupolar parameter    Cq=e*Q*V_zz/h
             eta: asymmetry parameters     (V_yy - V_xx)/ V_zz
             Q  : nuclear electric quadrupole moment in mb (millibarn)
            ----------------------------------------------------------------------
             ion       Cq(MHz)       eta       Q (mb)
            ----------------------------------------------------------------------
               1      -6.707       0.011      20.440
               2       0.000       0.149       0.000


        """

        # nuclear quadrupole moment for 14N
        line_index = match_lines(f, "ion       Cq(MHz)       eta       Q (mb)")
        if line_index:
            f.seek(line_index[-1], 0)
            f.readline()
            f.readline()
            quadrupole_moment = 4 / 5 * float(list(filter(None, f.readline().strip().split()))[1])
        else:
            quadrupole_moment = None

        """
         Total magnetic moment S=     2.0000002

         Fermi contact (isotropic) hyperfine coupling parameter (MHz)
         -------------------------------------------------------------
          ion      A_pw      A_1PS     A_1AE     A_1c      A_tot
         -------------------------------------------------------------
           1      -5.717    -5.714  -197.933    -4.206  -197.936
           2    1800.865  1813.132 14806.783 -2903.358 14794.516
           ...
         215      -0.000    -0.000     0.044     0.003     0.044
         -------------------------------------------------------------
        """
        f.seek(match_lines(f, "Fermi contact (isotropic) hyperfine coupling parameter (MHz)")[-1], 0)
        for _ in range(4):
            f.readline()
        fermi_contact = []
        while True:
            line = f.readline()
            if "--" in line:
                break
            else:
                fermi_contact.append(factor * float(list(filter(None, line.strip().split()))[-1]))
        """
         Dipolar hyperfine coupling parameters (MHz)
         ---------------------------------------------------------------------
          ion      A_xx      A_yy      A_zz      A_xy      A_xz      A_yz
         ---------------------------------------------------------------------
           1       2.633    -1.317    -1.316    12.990    12.990    11.208
           2    -112.810   157.716   -44.906 -2608.701  2489.345 -2644.609
        """
        f.seek(match_lines(f, "Dipolar hyperfine coupling parameters (MHz)")[-1], 0)
        for _ in range(4):
            f.readline()
        dipolar = []
        while True:
            line = f.readline()
            if "--" in line:
                break
            else:
                dipolar.append(list(map(lambda x: float(x) * factor, list(filter(None, line.strip().split()))[1:])))
        """
         Total hyperfine coupling parameters after diagonalization (MHz)
         (convention: |A_zz| > |A_xx| > |A_yy|)
         ----------------------------------------------------------------------
          ion      A_xx      A_yy      A_zz     asymmetry (A_yy - A_xx)/ A_zz
         ----------------------------------------------------------------------
           1    -210.399  -172.948  -210.461      -0.178
           2   12226.081 12196.944 19960.522      -0.001
        """

        return {"path": outcar_path, "energy": energy, "length_of_vectors": length_of_vectors,
                "quadrupole_moment": quadrupole_moment, "fermi_contact": fermi_contact,
                "dipolar": dipolar}


def construct_matrix(fermi_contact, dipolar):
    """
        dipolar is in order of A_xx      A_yy      A_zz      A_xy      A_xz      A_yz
    """

    Axx = dipolar[0] + fermi_contact
    Ayy = dipolar[1] + fermi_contact
    Azz = dipolar[2] + fermi_contact
    Axy = dipolar[3]
    Axz = dipolar[4]
    Ayz = dipolar[5]

    A_tensor = [[Axx, Axy, Axz],
                [Axy, Ayy, Ayz],
                [Axz, Ayz, Azz]]

    return A_tensor


def projection(hyperfine_tensor, vector):
    normed_vector = vector / np.linalg.norm(vector)
    return np.linalg.norm(normed_vector.dot(hyperfine_tensor)) * np.sign(hyperfine_tensor[0][0])


def read_dos(dos_path):
    # for phonon dos
    # assuming frequency in THz, and convert it to eV
    THz2eV = 0.00413566553853809
    frequency_list = []
    dos_list = []
    with open(dos_path, "r") as f:
        while True:
            line = f.readline()
            if line:
                if "#" in line:
                    continue
                else:
                    frequency, dos = list(filter(None, line.strip().split()))
                    frequency_list.append(float(frequency) * THz2eV)
                    dos_list.append(float(dos))
            else:
                break
    return {"frequency": frequency_list, "dos": dos_list}


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # read_phonon_from_outcar("OUTCAR_522", "522.json")
    # -------------------------------------------------------------------------
    # read_hyperfine_from_outcar("OUTCAR", "hyperfine.json", 1)
    # --------------------------------------------------------------------------

    # for directory in os.listdir():
    #     if os.path.isdir(directory) and directory[:2] == "x_" and not os.path.isfile(directory + "/OUTCAR"):
    #         print(directory + "/OUTCAR")
    # exit()
    # for frozen phonon
    # json.dump([read_hyperfine_from_outcar(directory + "/OUTCAR", 0.01) for directory in os.listdir()
    #            if os.path.isdir(directory) and directory[:2] == "x_"], open("fp.json", "w"))
    # for frozen phonon correction
    json.dump([read_hyperfine_from_outcar(directory + "/OUTCAR", 0.01) for directory in
               ["2_6_1.03/1/x_29_1.5", "2_6_0.99/1/x_29_1.5", "2_6_1.02/1/x_29_1.5", "2_6_1.005/1/x_29_1.5",
                "2_6_1.025/1/x_29_1.5", "2_6_1.015/1/x_29_1.5"]], open("corr.json", "w"))

    # for static
    # for directory in os.listdir():
    #     if os.path.isdir(directory) and not directory[0] == "2":
    #         json.dump([read_hyperfine_from_outcar(directory + "/OUTCAR", 0.01)],
    #                   open(f"a0_{round(float(directory), 6) + 1}.json", "w"))
    # ---------------------------------------------------------------------------------
    # test projection
    # hyperfine_dict = json.load(open("hyperfine.json", "r"))
    # fermi_contact = hyperfine_dict["fermi_contact"][0]
    # dipolar = hyperfine_dict["dipolar"][0]
    # hyperfine_tensor = construct_matrix(fermi_contact, dipolar)
    # vector = np.array([1, 1, 1])
    # vector = vector / np.linalg.norm(vector)
    # print(projection(hyperfine_tensor, vector))
    # -----------------------------------------------------------------------------
    # test last_line
    # with open("OUTCAR", "r") as f:
    #     for i in match_lines(f, "VOLUME and BASIS-vectors are now"):
    #         f.seek(i, 0)
    #         print(f.readline())
    # -----------------------------------------------------------------------------
    # generate band.yaml and read phonon from it
    # from scc_lib import linux_command
    # linux_command("phonopy --fc vasprun.xml;"
    #               "echo -e 'BAND = 0 0 0 0 0 0\nBAND_POINTS = 1\nDIM = 1 1 1\nEIGENVECTORS = .TRUE.\nFORCE_CONSTANTS = READ' > band.conf;"
    #               "phonopy band.conf;")
    # read_phonon_from_yaml("band.yaml", "vib.json")
    # --------------------------------------------------------------------------------
    # read_dos("total_dos.dat")
    # --------------------------------------------------------------------------------
    # check stray ones
    # mode_array = np.array(
    #     [[1, 16, 24.64934397], [1, 15, 24.63424756], [1, 68, 15.40916869], [1, 69, 15.40033787], [1, 29, 7.517594136],
    #      [1, 28, 7.432033558], [1, 65, 5.143546017], [1, 64, 5.121097039], [1, 63, -5.073905446], [1, 4, 3.580553992],
    #      [2, 62, 89.02755408], [2, 252, 70.61601229], [2, 63, 50.05923048], [2, 149, 47.92049877], [2, 67, 39.65455911],
    #      [2, 16, 39.02266452], [2, 31, 26.88189559], [2, 225, 23.46517707], [2, 3, 18.30726227], [2, 305, 17.70961047],
    #      [153, 62, 11.17828691], [153, 250, 3.80621702], [153, 15, -3.684418365], [153, 66, -3.477450617],
    #      [153, 64, 2.533720791], [153, 148, -2.071932528], [153, 16, 1.822514994], [153, 145, 1.798887934],
    #      [153, 63, 1.66078927], [153, 319, 1.45201262], [27, 15, -5.331141216], [27, 62, 5.121224339],
    #      [27, 69, -3.702139368], [27, 16, 3.608451348], [27, 250, 3.466318486], [27, 61, -2.947773533],
    #      [27, 1496, 2.357270976], [27, 146, -1.980507198], [27, 38, -1.851767676], [27, 66, -1.763415165],
    #      [232, 61, 15.50810399], [232, 252, 9.011452599], [232, 63, 8.449012255], [232, 62, -7.083964631],
    #      [232, 67, 4.955423373], [232, 16, -4.461263333], [232, 251, 4.222979322], [232, 149, 4.193543043],
    #      [232, 66, 3.478009114], [232, 225, 3.150301813], [231, 61, -5.867347323], [231, 69, 3.277926477],
    #      [231, 39, 2.587600365], [231, 4, 1.584116482], [231, 68, 1.486329327], [231, 66, 1.379516558],
    #      [231, 318, -1.354526492], [231, 95, 1.314355964], [231, 59, 1.200957598], [231, 13, 1.188105544]])
    #
    # mode_set = set(mode_array.T[1])
    # poscar_path_list = ["POSCAR"]
    # for dir in os.listdir():
    #     if os.path.isdir(dir) and dir[0] == "p" and int(dir.split("_")[0][1:]) in mode_set:
    #         poscar_path_list.append(dir + "/POSCAR")
