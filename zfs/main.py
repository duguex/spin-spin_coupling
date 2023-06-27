# cython:language_level=3

import logging
from time import time
import json
import numpy as np
# import pkg_resources
# from lxml import etree
from mpi4py import MPI
from numpy.fft import fftfreq, fftn
from scipy.constants import h
from scipy.constants import physical_constants

from parallel import ProcessorGrid, SymmetricDistributedMatrix

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s:%(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Energy
rydberg_to_ev = 13.606
rydberg_to_hartree = 0.5
hartree_to_rydberg = 2.0
hartree_to_ev = 27.212
joule_to_hz = 1. / h
hz_to_joule = h
joule_to_mhz = 1.E-6 / h
mhz_to_joule = 1.E6 * h
cminv_to_hz = 2.99793E10

# Length
bohr_to_angstrom = 0.52918
angstrom_to_bohr = 1.0 / bohr_to_angstrom
angstrom_to_cm = 1.0E-8
angstrom_to_m = 1.0E-10
bohr_to_m = bohr_to_angstrom * angstrom_to_m
m_to_bohr = 1.0 / bohr_to_m

# Time
fs_to_s = 1.0E-15


class ZFSCalculation:
    """Zero field splitting D tensor calculation.

    Generally, calculation of D tensor involves pairwise iteration over many wavefuctions
    (KS orbitals).

    Attributes:
        wf (Wavefunction): container for all KS orbitals
        cell (Poscar): defines cell size, R and G vectors
        ft (ft_grid): defines grid size for fourier transform

        ddig (ndarray): dipole-dipole interaction tensor in G space. Shape = (6, n1, n2, n3),
            where first index labels cartisian directions (xx, xy, xz, yy, yz, zz), last 3
            indices iterate over G space

        Iglobal (ndarray): global I array of shape (norbs, norbs, 6)
            first two indices iterate over wavefunctions, last index labels catesian directions
            in xx, xy, xz, yy, yz, xz manner
        I (ndarray): local I matrix, first two dimensions are distributed among processors
        D (ndarray): 3 by 3 matrix, total D tensor
        eigenvalue, eigenvector (ndarray): eigenvalues and eigenvectors of D tensor
        Dvalue, Evalue (float): scalar D and E parameters for triplet

    """

    def __init__(self, fpc, comm=MPI.COMM_WORLD):
        """Initialize ZFS calculation.

        Args:
            fpc (first-principles calculation): defines how
            comm (MPI.comm): MPI communicator on which ZFS calculation will be distributed.
        """

        # Define a 2D processor grid to parallelize summation over pairs of orbitals.
        self.processor_grid = ProcessorGrid(comm, square=True)
        if self.processor_grid.onroot:
            logger.info("zero field splitting calculation start...")
        self.processor_grid.print_info()

        # Parse wavefunctions, define cell and ft
        self.fpc = fpc

        # Declare ddig, I arrays and D arrays
        self.ddig = None

        if self.processor_grid.onroot:
            logger.info("Creating I array...")
        self.I = SymmetricDistributedMatrix(self.processor_grid,
                                            (self.fpc.nocc, self.fpc.nocc, 6), np.dtype(float))
        self.I.print_info("I")
        self.Iglobal = None

        self.D = np.zeros((3, 3))
        self.eigenvalue = np.zeros(3)
        self.eigenvector = np.zeros((3, 3))
        self.Dvalue = 0
        self.Evalue = 0

    def solve(self):
        """Compute and gather local block of I in each processor."""
        self.processor_grid.comm.barrier()
        tssolve = time()

        # Load wavefunctions from files
        orbital_list = set(list(range(self.I.mstart, self.I.mend)) + list(range(self.I.nstart, self.I.nend)))
        self.fpc.load(orbital_list=orbital_list, sdm=self.I)
        self.processor_grid.comm.barrier()

        # Compute dipole-dipole interaction tensor.
        # Due to symmetry we only need the upper triangular part of ddig.
        if self.processor_grid.onroot:
            logger.info("Computing dipole-dipole interaction tensor in G space...")

        ddig = compute_ddi_g(self.fpc.cell, self.fpc.wf.fft_grid)
        self.ddig = ddig[np.triu_indices(3)]

        # Compute contribution to D tensor from every pair of electrons
        self.processor_grid.comm.barrier()
        if self.processor_grid.onroot:
            logger.info("Iterating over pairs...")

        timer = Timer(len(list(self.I.get_triu_iterator())), pace=0.01)

        for local_orbital_i, local_orbital_j in self.I.get_triu_iterator():
            # Load two wavefunctions
            global_orbital_i, global_orbital_j = self.I.ltog(local_orbital_i, local_orbital_j)
            if global_orbital_i == global_orbital_j:
                timer.count()
                continue  # skip diagonal terms
            if self.fpc.occ_spinband_map[global_orbital_i][0] == \
                    self.fpc.occ_spinband_map[global_orbital_j][0]:
                # parallel
                chi = 1
            else:
                # antiparallel
                chi = -1

            psi_r_i = self.fpc.get_psi_r(global_orbital_i)
            psi_r_j = self.fpc.get_psi_r(global_orbital_j)
            rho_g_i = self.fpc.get_rho_g(global_orbital_i)
            rho_g_j = self.fpc.get_rho_g(global_orbital_j)
            rho_g = compute_rho_g(psi_r_i, psi_r_j, self.fpc.wf.fft_grid, rho_g_i=rho_g_i, rho_g_j=rho_g_j)

            # Factor to be multiplied with I:
            #   chi comes from spin direction
            #   prefactor comes from physical constants and unit conversions'
            #   omega**2 comes from convention of FT used here

            factor = chi * prefactor * self.fpc.cell.omega ** 2
            local_I = factor * np.tensordot(self.ddig, rho_g, axes=3)
            # TODO: check if it is safe to use the real part only
            # assert np.all(local_I.imag == 0)
            self.I[local_orbital_i, local_orbital_j, ...] = local_I.real

            timer.count()

        self.I.symmetrize()

        # All processor sync local matrix to get global matrix
        self.Iglobal = self.I.collect()

        # Sum over G vectors to get D tensor
        self.D[np.triu_indices(3)] = np.sum(self.Iglobal, axis=(0, 1))
        self.D = self.D + self.D.T - np.diag(self.D.diagonal())
        self.eigenvalue, self.eigenvector = np.linalg.eig(self.D)

        # For triplet states, compute D and E parameters:
        # Denote three eigenvalues as Dx, Dy, Dz: |Dz| > |Dx| > |Dy|
        # D = 3/2 Dz, E = 1/2(Dx - Dy)
        args = np.abs(self.eigenvalue).argsort()
        dy, dx, dz = self.eigenvalue[args]
        self.Dvalue = 1.5 * dz
        self.Evalue = 0.5 * (dx - dy)

        if self.processor_grid.onroot:
            logger.info(f"Total D tensor (MHz): {self.D}")
            logger.info(f"D eigenvalues (MHz): {self.eigenvalue}")
            logger.info(f"D eigenvectors: {self.eigenvector}")
            logger.info(f"Dx, Dy, Dz (|Dz| > |Dx| > |Dy|) (MHz): {dx}, {dy}, {dz}")
            logger.info(f"Scalar D = {self.Dvalue:.2f} MHz, E = {self.Evalue:.2f} MHz")

            zfs_dump = {"D tensor": self.D.tolist(),
                        "eigenvalues": self.eigenvalue.tolist(),
                        "eigenvectors": self.eigenvector.tolist(),
                        "Dx": dx,
                        "Dy": dy,
                        "Dz": dz,
                        "D": self.Dvalue,
                        "E": self.Evalue}
            json.dump(zfs_dump, open("zfs.json", "w"), indent=4)
            logger.info(f"Time elapsed for pair iteration: {time() - tssolve:.0f}s")


class Timer(object):
    def __init__(self, total_count, pace=0.1, comm=MPI.COMM_WORLD):
        self.total_count = total_count
        self.pace = pace
        self.threshold = total_count * pace
        self.comm = comm
        self.onroot = self.comm.Get_rank() == 0

        self.total_counter = 0
        self.counter = 0
        self.start_time = time()

    def count(self):
        self.total_counter += 1
        self.counter += 1

        if self.counter >= self.threshold:
            self.counter = 0
            current_time = time()
            if self.onroot:
                logger.info(f"{self.total_counter} pairs as "
                            f"{(100 * self.total_counter) // self.total_count} % finished in "
                            f"{round(current_time - self.start_time, 2)} s")
            self.start_time = current_time


# Define functions to compute dipole-dipole interactions (abbr. "ddi")
# The dipole-dipole interaction is a 3 by 3 tensor (with unit bohr^-3)
# as a function of r (labeled as ddir) or G (labeled as ddig)
# ddir is defined as eq. 4 in PRB 77, 035119 (2008), without the leading 1/2
# ddig can be computed by Fourier transform ddir, or computed analytically in G space


def compute_ddi_g(cell, ft):
    """Compute dipole-dipole interaction in G space.
    ddi(G)_{ab} = 4 * pi * [ Ga * Gb / G^2 - delta(a,b) / 3 ]
    see eq. 17 PRB 77, 035119 (2008)
    a, b are cartesian indices.

    Args:
        cell (Poscar or etc.): Cell on which to compute ddi_g.
        ft (ft_grid): FT which defines grid size.

    Returns:
        np.ndarray of shape (3, 3, np.prod(ft), np.prod(ft)). First two indices iterate
            over cartesian coordinates, last two indices iterate over G space.
    """

    n1, n2, n3 = ft
    G1, G2, G3 = cell.G1, cell.G2, cell.G3
    omega = cell.omega

    ddi_g = np.zeros([3, 3, n1, n2, n3])

    G1_arr = np.outer(G1, fftfreq(n1, d=1 / n1))
    G2_arr = np.outer(G2, fftfreq(n2, d=1 / n2))
    G3_arr = np.outer(G3, fftfreq(n3, d=1 / n3))

    Gx = (G1_arr[0, :, np.newaxis, np.newaxis] +
          G2_arr[0, np.newaxis, :, np.newaxis] +
          G3_arr[0, np.newaxis, np.newaxis, :])
    Gy = (G1_arr[1, :, np.newaxis, np.newaxis] +
          G2_arr[1, np.newaxis, :, np.newaxis] +
          G3_arr[1, np.newaxis, np.newaxis, :])
    Gz = (G1_arr[2, :, np.newaxis, np.newaxis] +
          G2_arr[2, np.newaxis, :, np.newaxis] +
          G3_arr[2, np.newaxis, np.newaxis, :])

    Gxx = Gx ** 2
    Gyy = Gy ** 2
    Gzz = Gz ** 2
    Gxy = Gx * Gy
    Gxz = Gx * Gz
    Gyz = Gy * Gz
    Gsquare = Gxx + Gyy + Gzz
    Gsquare[0, 0, 0] = 1  # avoid runtime error message, G = 0 term will be excluded later

    ddi_g[0, 0, ...] = Gxx / Gsquare - 1. / 3.
    ddi_g[1, 1, ...] = Gyy / Gsquare - 1. / 3.
    ddi_g[2, 2, ...] = Gzz / Gsquare - 1. / 3.
    ddi_g[0, 1, ...] = ddi_g[1, 0, ...] = Gxy / Gsquare
    ddi_g[0, 2, ...] = ddi_g[2, 0, ...] = Gxz / Gsquare
    ddi_g[1, 2, ...] = ddi_g[2, 1, ...] = Gyz / Gsquare

    ddi_g[..., 0, 0, 0] = 0
    ddi_g *= 4 * np.pi / omega

    return ddi_g


def compute_ddir(cell, ft):
    """Compute dipole-dipole interaction in R space.

    ddi(r)_{ab} = ( r^2 * delta(a,b) - 3 * ra * rb ) / r^5
    a, b are cartesian indices.

    Args:
        cell (Poscar or etc.): Cell on which to compute ddig.
        ft (ft_grid): FT which defines grid size.

    Returns:
        np.ndarray of shape (3, 3, np.prod(ft), np.prod(ft)). First two indices iterate
            over cartesian coordinates, last two indices iterate over R space.
    """

    n1, n2, n3 = ft
    R1, R2, R3 = cell.R1, cell.R2, cell.R3

    ddir = np.zeros([3, 3, n1, n2, n3])

    for ir1, ir2, ir3 in np.ndindex(n1, n2, n3):
        if ir1 == ir2 == ir3 == 0:
            continue  # neglect r = 0 component

        r = ((ir1 - n1 * int(ir1 > n1 / 2)) * R1 / n1 +
             (ir2 - n2 * int(ir2 > n2 / 2)) * R2 / n2 +
             (ir3 - n3 * int(ir3 > n3 / 2)) * R3 / n3)

        rnorm = np.linalg.norm(r)
        ddir[..., ir1, ir2, ir3] = (rnorm ** 2 * np.eye(3) - 3 * np.outer(r, r)) / rnorm ** 5
    return ddir


def compute_rho_g(psi_r_i, psi_r_j, ft, rho_g_i=None, rho_g_j=None):
    """Compute rho(G, -G) for two electrons occupying two (KS) orbitals.

    rho(G, -G) is defined as f1(G) * f2(-G) - |f3(G)|^2,
    which is equal to f1(G) * conj(f2(G)) - f3(G) * conj(f3(G))
    f1, f2 and f3 are defined following PRB 77, 035119 (2008):
      f1(r) = |psi(r)_i|^2
      f2(r) = |psi(r)_j|^2
      f3(r) = conj(psi1(r)) * psi2(r)
    f1(G), f2(G) and f3(G) are obtained by Fourier Transform of f1(r), f2(r) and f3(r)
    rho(r) is computed for debug purpose as inverse FT of rho(G, -G) and returned as well

    Args:
        psi_r_1, psi_r_2 (np.ndarray): R space wavefunction for electron 1 and 2.
        ft (ft_grid): FT which defines grid size.
        rho_g_1, rho_g_2 (np.ndarray): R space charge density for electron 1 and 2.
            If not provided, will be computed from psi_r_1 and psi_r_2.

    Returns:
        rho(G, -G) as a np.ndarray of shape (np.prod(ft), np.prod(ft)).

    """

    if rho_g_i is not None:
        assert rho_g_i.shape == psi_r_i.shape
        f_g_i = rho_g_i
    else:
        f_r_i = psi_r_i * np.conj(psi_r_i)
        assert f_r_i.ndim == 3 and np.all(f_r_i.shape == ft)
        f_g_i = fftn(f_r_i) / np.prod(ft)

    if rho_g_j is not None:
        assert rho_g_j.shape == psi_r_j.shape
        f_g_j = rho_g_j
    else:
        f_r_j = psi_r_j * np.conj(psi_r_j)
        assert f_r_j.ndim == 3 and np.all(f_r_j.shape == ft)
        f_g_j = fftn(f_r_j) / np.prod(ft)

    f_r_ij = psi_r_i * np.conj(psi_r_j)
    assert f_r_ij.ndim == 3 and np.all(f_r_ij.shape == ft)
    f_g_ij = fftn(f_r_ij) / np.prod(ft)

    # rho_j = f_g_i * np.conj(f_g_j)
    # rho_k = f_g_ij * np.conj(f_g_ij)

    rho_g = f_g_i * np.conj(f_g_j) - f_g_ij * np.conj(f_g_ij)
    # assert rho_g.ndim == 3 and np.all(rho_g.shape == ft)
    # rho_r = ifftn(rho_g) * np.prod(ft)

    return rho_g  # , rho_r, rho_j, rho_k


def compute_delta_model_rho_g(cell, ft, d1, d2, d3, s=1):
    """Compute rho(G, -G) for two point dipoles.

    Two spin dipoles are approximated homogeneious dipole gas in small boxes

    Args:
        cell (Poscar or etc.): Cell on which to compute ddig.
        ft (..common.ft.FourierTransform): FT which defines grid size.
        d1, d2, d3 (float): distance between two dipoles in 3 dimensions
        s (float): box size

    Returns:
        rho(G, -G) as a np.ndarray of shape (np.prod(ft), np.prod(ft))
    """

    n1, n2, n3 = ft
    N = np.prod(ft)
    R1, R2, R3 = cell.R1, cell.R2, cell.R3
    omega = cell.omega

    ns1 = int(n1 * s / R1[0])
    ns2 = int(n2 * s / R2[1])
    ns3 = int(n3 * s / R3[2])

    nd1 = int(n1 * d1 / R1[0])
    nd2 = int(n2 * d2 / R2[1])
    nd3 = int(n3 * d3 / R3[2])

    print(ns1, ns2, ns3)
    print(nd1, nd2, nd3)
    print("effective d1, d2, d3: ", nd1 * R1[0] / n1, nd2 * R2[1] / n2, nd3 * R3[2] / n3)

    psi_r_1 = np.zeros([n1, n2, n3])
    psi_r_2 = np.zeros([n1, n2, n3])

    for ir1, ir2, ir3 in np.ndindex(ns1, ns2, ns3):
        psi_r_1[ir1, ir2, ir3] = 1.
        psi_r_2[nd1 + ir1, nd2 + ir2, nd3 + ir3] = 1.

    psi_r_1 /= np.sqrt(np.sum(psi_r_1 ** 2) * omega / N)
    psi_r_2 /= np.sqrt(np.sum(psi_r_2 ** 2) * omega / N)

    return compute_rho_g(psi_r_1, psi_r_2, ft)

    # rho_g, rho_r, rho_j, rho_k = compute_rho_g(psi_r_1, psi_r_2, ft)
    # return rho_g, rho_r, rho_j, rho_k, psi_r_1, psi_r_2


gamma = physical_constants["electron gyromag. ratio"][0]
hbar = physical_constants["Planck constant over 2 pi"][0]
mu0 = physical_constants["mag. constant"][0]
ge = physical_constants["electron g factor"][0]
mub = physical_constants["Bohr magneton"][0]

prefactor = np.prod(
    [
        # -1,  # sign convention for D tensor
        # 1. / 2,            # eq. 2 from PRB paper
        1. / 4,  # eq. 2 and eq. 8 from PRB paper
        mu0 / (4 * np.pi),  # magnetic constant
        (gamma * hbar) ** 2,  # conversion factor from unitless spin to magnetic moment

        # at this point, unit is J m^3
        m_to_bohr ** 3,
        joule_to_mhz,
        # at this point, unit is MHz bohr^3
    ]
)
