# cython:language_level=3

import logging

import numpy as np
import json
from parallel import mpiroot
from poscar import Poscar
from wavecar import Wavecar

logger = logging.getLogger("root")


class Vasp:
    """
    Attributes:
        orbital_count (int): total number of KS orbitals to be considered
        up_count/down_count (int): number of spin up/down orbitals
        spinband_orbital_map (dict): (spin, band index) -> orb index map
        orbital_spinband_map (list): orb index -> (spin, band index) map
        orbital_psi_r_map (dict): orb index -> orb object (3D array) map

        cell (Poscar): defines cell size, R and G vectors
        ft (FourierTransform): defines grid size for fourier transform

        an example of Wavecar class

        'path': 'WAVECAR_std',
        'soc': False,
        'gamma': False,
        'gamma_half': 'x',
        'wavefunction': < _io.BufferedReader name = 'WAVECAR_std' >,
        'record_length': 178336,
        'nspin': 2,
        'rtag': 45200,
        'wfprec': <class 'numpy.complex64'>,
        'nkpoints': 8,
        'nbands': 540,
        'encut': 400.0,
        'omega': 1225.7049639242346,
        'planewave_grid': array([37, 37, 37]),
        'fft_grid': array([37, 37, 37]),
        'nplanewaves': array([22155, 22282, 22282, 22268, 22282, 22268, 22268, 22292]),
        'cell': array([[1.07019289e+01, 4.60841571e-03, 4.60841571e-03],
        'reciprocal': array([[9.34411337e-02, -4.02198745e-05, -4.02198745e-05],
        'kpoint_vectors': array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        'bands': array([[[-11.88073433, -11.05800703, -10.86972621, ..., 17.82546246, 17.83592533, 17.85936261],
        'occupations': array([[[1., 1., 1., ..., 0., 0., 0.],
    """

    def __init__(self, poscar_path="POSCAR", wavecar_path="WAVECAR",
                 gamma=True, soc=False, kpoint_index=1, grid_scale=1):

        if gamma==True:
            assert kpoint_index==1, "kpoint_index must be 1 in gamma mode"

        self.cell = Poscar(poscar_path)
        self.wf = Wavecar(path=wavecar_path, soc=soc, gamma=gamma, grid_scale=grid_scale)
        self.kpoint_index = kpoint_index
        # TODO(lmz): test other kpoints exculde gamma
        assert self.wf.nspin == 2, "only for single kpoint and spin polarized currently."

        # Get band indices (starting from 1) witt significant occupations
        up_occ_list = np.where(self.wf.occupations[0, 0] > 0.8)[0] + 1
        down_occ_list = np.where(self.wf.occupations[1, 0] > 0.8)[0] + 1

        nup_occ = len(up_occ_list)
        ndown_occ = len(down_occ_list)
        self.nocc = nup_occ + ndown_occ

        occ_spinband_map2 = [("up", up_occ_list[orbital_index]) if orbital_index < nup_occ else
                             ("down", down_occ_list[orbital_index - nup_occ]) for orbital_index in range(self.nocc)]
        self.occ_spinband_map = [("up", up_occ_list[orbital_index]) for orbital_index in range(nup_occ)]
        self.occ_spinband_map += [("down", down_occ_list[orbital_index]) for orbital_index in range(ndown_occ)]
        # spinband_occ_map = {self.occ_spinband_map[orbital]: orbital for orbital in range(self.nocc)}

        assert self.occ_spinband_map == occ_spinband_map2, "map1 and map2 should be the same"

        # orbital_psi_g_arr_map = {}
        self.orbital_psi_r_map = {}
        self.orbital_rho_g_map = {}

        if mpiroot:
            logger.info(f"system name: {self.cell.name}")
            logger.info(f"occupied up/down orbitals = {nup_occ}/{ndown_occ}")
            logger.info(f"fft grid: {self.wf.fft_grid}")

    def load(self, orbital_list, sdm=None):
        """Load read space KS orbitals to memory, store in wfc.iorb_psir_map.

        Args:
            orbital_list: a list of integers representing orbital indices.
            sdm: a SymmetricDistributedMatrix object indicating how the wavefunction is distributed.

        Returns:
            After load is called, the wavefunction will be loaded into self.wfc.
        """
        if mpiroot:
            logger.info(f"{self.__class__.__name__}: loading orbitals into memory...")

        for orbital_index in orbital_list:
            spin, band = self.occ_spinband_map[orbital_index]
            # TODO(lmz): kr_phase gamma
            psi_r = self.wf.wfc_r(spin_index=1 if spin == "up" else 2,
                                  band_index=band,
                                  kpoint_index=self.kpoint_index,
                                  kr_phase=True)

            psi_r = self.normalize(psi_r)

            # write into list self.orbital_psi_r_map
            self.orbital_psi_r_map[orbital_index] = psi_r

            # write into list self.orbital_rho_g_map
            rho_r = psi_r * np.conj(psi_r)
            self.orbital_rho_g_map[orbital_index] = np.fft.fftn(rho_r) / np.prod(self.wf.fft_grid)

    # def set_psi_g_arr(self, orbital_index, psi_g_arr):
    #     if orbital_index in self.orbital_psi_g_arr_map:
    #         raise ValueError("psi_g_arr {} already set".format(orbital_index))
    #     self.orbital_psi_g_arr_map[orbital_index] = psi_g_arr

    def normalize(self, psi_r):
        """Normalize psi_r."""
        assert np.all(psi_r.shape == self.wf.fft_grid)
        norm = np.sqrt(np.sum(np.abs(psi_r) ** 2) * self.cell.omega / np.prod(self.wf.fft_grid))
        return psi_r / norm

    def get_psi_r(self, orbital_index):
        """Get psi(r) of certain index"""
        return self.orbital_psi_r_map[orbital_index]

    def get_rho_g(self, orbital_index):
        """Get rho(G) of certain index"""
        return self.orbital_rho_g_map[orbital_index]


if __name__ == "__main__":
    vasp = Vasp(poscar_path="CONTCAR", wavecar_path="WAVECAR",
                gamma=True, soc=False, grid_scale=1)
