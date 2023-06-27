# cython:language_level=3
# please run python with python -O to
import argparse
import os

from main import ZFSCalculation
from vasp import Vasp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--poscar", default="CONTCAR", type=str,
                        help="The path of POSCAR or CONTCAR file. "
                             "Default is CONTCAR.")
    parser.add_argument("-w", "--wavecar", default="WAVECAR", type=str,
                        help="The path of WAVECAR file. "
                             "Default is WAVECAR.")
    parser.add_argument("-k", "--kpoint", default=1, type=int,
                        help="The index of kpoint in Brillouin zone sampling."
                             "Default is 1.")
    parser.add_argument("-s", "--scale", default=1, type=int,
                        help="How many times are the 3D grid used in Fourier Transform"
                             "compared to the mesh grid of the plane-wave in WAVECAR file."
                             "Default is 1.")
    parser.add_argument("-g", "--gamma", action="store_true",
                        help="True if vasp_gam else False")

    args = parser.parse_args()
    poscar = args.poscar
    wavecar = args.wavecar
    kpoint = args.kpoint
    grid_scale = args.scale
    gamma = args.gamma

    print(f"poscar: {poscar}\n"
          f"wavecar: {wavecar}\n"
          f"kpoint: {kpoint}\n"
          f"grid_scale: {grid_scale}\n"
          f"gamma: {gamma}")

    if not "/" in wavecar:
        path = "."
    else:
        path = os.path.split(wavecar)[0]

    poscar = os.path.abspath(poscar)
    wavecar = os.path.abspath(wavecar)

    os.chdir(path)
    vasp = Vasp(poscar_path=poscar, wavecar_path=wavecar, kpoint_index=kpoint,
                gamma=gamma, soc=False, grid_scale=grid_scale)
    ZFSCalculation(fpc=vasp).solve()

# bsub -q ckduan -n 24 -o %J.log mpirun python -O run.py -p CONTCAR -w WAVECAR -k 1 -s 1 -g
