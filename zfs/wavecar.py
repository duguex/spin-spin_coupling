# cython:language_level=3

import logging

import numpy as np

logger = logging.getLogger("root")

#  Some important Parameters, to convert to a.u.
#  - au2angstrom     =  1. a.u. in Angstrom
#  - Ry2eV    =  1 Ry in Ev
#  - EVTOJ     =  1 eV in Joule
#  - AMTOKG    =  1 atomic mass unit ("proton mass") in kg
#  - BOLKEV    =  Boltzmanns constant in eV/K
#  - BOLK      =  Boltzmanns constant in Joule/K

au2angstrom = 0.529177249
# 0.5 Hartree
Ry2eV = 13.605826
# CLIGHT   =  137.037          # speed of light in a.u.
# EVTOJ    = 1.60217733E-19
# AMTOKG   = 1.6605402E-27
# BOLKEV   = 8.6173857E-5
# BOLK     = BOLKEV * EVTOJ
# EVTOKCAL = 23.06

# FELECT    =  (the electronic charge)/(4*pi*the permittivity of free space)
#         in atomic units this is just e^2
# EDEPS    =  electron charge divided by the permittivity of free space
#         in atomic units this is just 4 pi e^2
# HSQDTM    =  (plancks CONSTANT/(2*np.pi))**2/(2*ELECTRON MASS)
#
# CI2 * np.pi  = 1j * 2 * np.pi
# FELECT = 2 * au2angstrom * Ry2eV
# EDEPS  = 4 * np.pi * 2 * Ry2eV * au2angstrom
HSQDTM = Ry2eV * au2angstrom * au2angstrom

# vector field A times momentum times e/ (2 m_e c) is an energy
# magnetic moments are supplied in Bohr magnetons
# e / (2 m_e c) A(r) p(r)    =  energy
# e / (2 m_e c) m_s x ( r - r_s) / (r-r_s)^3 hbar nabla    = 
# e^2 hbar^2 / (2 m_e^2 c^2) 1/ lenght^3    =  energy
# conversion factor from magnetic moment to energy
# checked independently in SI by Gilles de Wijs

# MAGMOMTOENERGY  = 1 / CLIGHT**2 * au2angstrom**3 * Ry2eV

# dimensionless number connecting input and output magnetic moments
# au2angstrom e^2 (2 m_e c^2)

# MOMTOMOM   = au2angstrom / CLIGHT / CLIGHT / 2
# au2angstrom2   = au2angstrom * au2angstrom
# au2angstrom3   = au2angstrom2 * au2angstrom
# au2angstrom4   = au2angstrom2 * au2angstrom2
# au2angstrom5   = au2angstrom3 * au2angstrom2

# dipole moment in atomic units to Debye
au2Debye = 2.541746


class Wavecar(object):
    """
    This program is motivated by PIESTA written by Ren Hao <renh@upc.edu.cn>.

    The format of VASP WAVECAR, as shown in
        http://www.andrew.cmu.edu/user/feenstra/wavetrans/
    is:
        Record-length #spin components RTAG(a value specifying the precision)
        #k-points #bands ENCUT(maximum energy for plane_waves)
        LatVec-A
        LatVec-B
        LatVec-C
        Loop over spin
           Loop over k-points
              #plane_waves, k vector
              Loop over bands
                 band energy, band occupation
              End loop over bands
              Loop over bands
                 Loop over plane_waves
                    Plane-wave coefficient
                 End loop over plane_waves
              End loop over bands
           End loop over k-points
        End loop over spin
    """

    def __init__(self, path, soc=False, gamma=False, gamma_half='x', grid_scale=1):

        self.path = path
        self.soc = soc
        self.gamma = gamma
        self.gamma_half = gamma_half.lower()

        assert not (soc and gamma), 'The two settings conflict!'
        assert self.gamma_half in ['x', 'z'], 'Gamma_half must be "x" or "z"'

        self.wavefunction = open(self.path, 'rb')

        # read the basic information
        '''
        Read the system information from WAVECAR, which is written in the first
        two record.

        rec1: record_length, nspin, rtag
        rec2: nkpoints, nbands, encut, ((cell(i,j) i=1, 3), j=1, 3)
        '''

        # goto the start of the file and read the first record
        self.wavefunction.seek(0)
        self.record_length, self.nspin, self.rtag = np.array(np.fromfile(self.wavefunction, dtype=float, count=3),
                                                             dtype=np.int64)
        if self.soc:
            assert self.nspin == 1, "NSPIN = 1 for noncollinear version WAVECAR!"

        logger.debug(f"record_length {self.record_length}\n"
                     f"self.nspin {self.nspin}\n"
                     f"self.rtag {self.rtag}")
        '''
        Set wavefunction coefficients precision:
            TAG = 45200: single precision complex, np.complex64, or complex(qs)
            TAG = 45210: double precision complex, np.complex128, or complex(q)
        '''
        if self.rtag == 45200:
            self.wfprec = np.complex64
        elif self.rtag == 45210:
            self.wfprec = np.complex128
        elif self.rtag == 53300:
            raise ValueError("VASP5 WAVECAR format, not implemented yet")
        elif self.rtag == 53310:
            raise ValueError("VASP5 WAVECAR format with double precision coefficients, not implemented yet")
        else:
            raise ValueError("Invalid TAG values: {}".format(self.rtag))

        # the second record
        self.wavefunction.seek(self.record_length)
        dump = np.fromfile(self.wavefunction, dtype=float, count=12)

        self.nkpoints = int(dump[0])  # No. of k-points
        self.nbands = int(dump[1])  # No. of bands
        self.encut = dump[2]  # Energy cutoff
        # real space supercell basis
        self.cell = dump[3:].reshape((3, 3))
        # real space supercell volume
        self.omega = np.linalg.det(self.cell)
        # reciprocal space supercell basis
        self.reciprocal = np.linalg.inv(self.cell).T
        logger.debug(f"kpoints {self.nkpoints}\n"
                     f"nbands {self.nbands}\n"
                     f"encut {self.encut}\n"
                     f"cell {self.cell}\n"
                     f"volume {self.omega}\n"
                     f"reciprocal {self.reciprocal}")

        # Minimum FFT grid size
        grid_point = np.ceil(np.sqrt(self.encut / Ry2eV) /
                             (2 * np.pi / (np.linalg.norm(self.cell, axis=1) / au2angstrom)))
        self.planewave_grid = np.array(2 * grid_point + 1, dtype=int)
        self.fft_grid = self.planewave_grid * grid_scale
        # read the band information
        '''
        Extract KS energies and Fermi occupations from WAVECAR.
        '''

        self.nplanewaves = np.zeros(self.nkpoints, dtype=int)
        self.kpoint_vectors = np.zeros((self.nkpoints, 3), dtype=float)
        self.bands = np.zeros((self.nspin, self.nkpoints, self.nbands), dtype=float)
        self.occupations = np.zeros((self.nspin, self.nkpoints, self.nbands), dtype=float)
        '''
        Loop over spin
           Loop over k-points
           
              number of plane_waves, k vector
              Loop over bands
                 band energy, band occupation
              End loop over bands
              
              Loop over bands
                 Loop over plane_waves
                    Plane-wave coefficient
                 End loop over plane_waves
              End loop over bands
              
           End loop over k-points
        End loop over spin
        '''
        for spin_index in range(self.nspin):
            for kpoint_index in range(self.nkpoints):
                block_index = self.get_block_index(spin_index + 1, kpoint_index + 1, 1) - 1
                self.wavefunction.seek(block_index * self.record_length)
                dump = np.fromfile(self.wavefunction, dtype=float, count=4 + 3 * self.nbands)
                if spin_index == 0:
                    self.nplanewaves[kpoint_index] = int(dump[0])
                    self.kpoint_vectors[kpoint_index] = dump[1:4]
                dump = dump[4:].reshape((-1, 3))
                self.bands[spin_index, kpoint_index, :] = dump[:, 0]
                self.occupations[spin_index, kpoint_index, :] = dump[:, 2]

        #
        # if self.nkpoints > 1:
        #     # the length of kpath
        #     kpath_length = np.linalg.norm(np.dot(np.diff(self.kpoint_vectors, axis=0), self.reciprocal), axis=1)
        #     self.kpath = np.concatenate(([0, ], np.cumsum(kpath_length)))
        # else:
        #     self.kpath = None

    def read_coef(self, spin_index=1, kpoint_index=1, band_index=1, norm=False):
        """
        Read the planewave coefficients of specified KS states.
        """

        self.checkIndex(spin_index, kpoint_index, band_index)
        block_index = self.get_block_index(spin_index, kpoint_index, band_index)
        self.wavefunction.seek(block_index * self.record_length)
        nplane_waves = self.nplanewaves[kpoint_index - 1]
        dump = np.fromfile(self.wavefunction, dtype=self.wfprec, count=nplane_waves)
        coef = np.asarray(dump, dtype=np.complex128)
        if norm:
            coef /= np.linalg.norm(coef)
        return coef

    def checkIndex(self, spin_index, kpoint_index, band_index):
        """
        Check if the index is valid!
        """
        assert 1 <= spin_index <= self.nspin, 'Invalid spin index!'
        assert 1 <= kpoint_index <= self.nkpoints, 'Invalid kpoint index!'
        assert 1 <= band_index <= self.nbands, 'Invalid band index!'

    def get_block_index(self, spin_index=1, kpoint_index=1, band_index=1):
        """
        Return the rec position for specified KS state.
        """

        self.checkIndex(spin_index, kpoint_index, band_index)
        rec = 2 + (spin_index - 1) * self.nkpoints * (self.nbands + 1) + \
              (kpoint_index - 1) * (self.nbands + 1) + band_index
        return rec

    # def get_kpath(self, nkseg=None):
    #     '''
    #     Construct k-point path, find out the k-path boundary if possible.
    #
    #     nkseg is the number of k-points in each k-path segments.
    #     '''
    #
    #     if nkseg is None:
    #         kpoints_path = os.path.dirname(self.path) + "/KPOINTS"
    #         if os.path.isfile(kpoints_path):
    #             kfile = open(kpoints_path).readlines()
    #             if kfile[2][0].upper() == 'L':
    #                 nkseg = int(kfile[1].split()[0])
    #             else:
    #                 raise ValueError(
    #                     'Error reading number of k-points from KPOINTS')
    #     assert nkseg > 0
    #
    #     nsec = self.nkpoints // nkseg
    #
    #     v = self.kpoint_vectors.copy()
    #     for ii in range(nsec):
    #         ki = ii * nkseg
    #         kj = (ii + 1) * nkseg
    #         v[ki:kj, :] -= v[ki]
    #
    #     self.kpath = np.linalg.norm(np.dot(v, self.reciprocal), axis=1)
    #     for ii in range(1, nsec):
    #         ki = ii * nkseg
    #         kj = (ii + 1) * nkseg
    #         self.kpath[ki:kj] += self.kpath[ki - 1]
    #
    #         self.kbound = np.concatenate(
    #             (self.kpath[0::nkseg], [self.kpath[-1], ]))
    #
    #     return self.kpath, self.kbound

    def gvectors(self, kpoint_index=1, force_Gamma=False, check_consistency=True):
        """
        Generate the G-vectors that satisfies the following relation
            (G + k)**2 / 2 < ENCUT
        """
        assert 1 <= kpoint_index <= self.nkpoints, 'Invalid kpoint index!'

        kvector = self.kpoint_vectors[kpoint_index - 1]
        # force_Gamma: consider gamma-only case regardless of the actual setting
        gamma = True if force_Gamma else self.gamma

        fx, fy, fz = [np.arange(n, dtype=int) for n in self.planewave_grid]
        fx[self.planewave_grid[0] // 2 + 1:] -= self.planewave_grid[0]
        fy[self.planewave_grid[1] // 2 + 1:] -= self.planewave_grid[1]
        fz[self.planewave_grid[2] // 2 + 1:] -= self.planewave_grid[2]
        if gamma:
            if self.gamma_half == 'x':
                fx = fx[:self.planewave_grid[0] // 2 + 1]
            else:
                fz = fz[:self.planewave_grid[2] // 2 + 1]

        # In meshgrid, fx run the fastest, fz the slowest
        gz, gy, gx = np.array(np.meshgrid(fz, fy, fx, indexing='ij')).reshape((3, -1))
        kgrid = np.array([gx, gy, gz], dtype=float).T
        if gamma:
            # parallel gamma version of VASP WAVECAR exclude some planewave components, -DwNGZHalf
            if self.gamma_half == 'x':
                kgrid = kgrid[(gx > 0) | ((gx == 0) & (gy > 0)) | ((gx == 0) & (gy == 0) & (gz >= 0))]
            else:
                kgrid = kgrid[(gz > 0) | ((gz == 0) & (gy > 0)) | ((gz == 0) & (gy == 0) & (gx >= 0))]

        # Kinetic_Energy = (G + k)**2 / 2
        # HSQDTM = hbar**2/(2*ELECTRON MASS)
        KENERGY = HSQDTM * np.linalg.norm(np.dot(kgrid + kvector[np.newaxis, :], 2 * np.pi * self.reciprocal),
                                          axis=1) ** 2

        # find Gvectors where (G + k)**2 / 2 < ENCUT
        Gvector = kgrid[np.where(KENERGY < self.encut)[0]]
        # logger.debug(f"number of grid {kgrid.shape} reduced by encut {Gvector.shape}")

        # Check if the calculated number of planewaves and the one recorded in the WAVECAR are equal

        if check_consistency:
            if Gvector.shape[0] != self.nplanewaves[kpoint_index - 1]:
                if Gvector.shape[0] * 2 == self.nplanewaves[kpoint_index - 1]:
                    if not self.soc:
                        raise ValueError('''
                        It seems that you are reading a WAVECAR from a NONCOLLINEAR VASP.
                        Please set 'soc = True' when loading the WAVECAR.
                        For example:

                            wfc = Wavecar('WAVECAR', soc=True)
                        ''')
                elif Gvector.shape[0] == 2 * self.nplanewaves[kpoint_index - 1] - 1:
                    if not self.gamma:
                        raise ValueError('''
                        It seems that you are reading a WAVECAR from a GAMMA-ONLY VASP.  Please set
                        'gamma = True' when loading the WAVECAR.  Moreover, you may want to set
                        "gamma_half" if you are using VASP version <= 5.2.x.  For VASP <= 5.2.x, check
                        which FFT VASP uses by the following command:

                            $ grep 'use.* FFT for wave' OUTCAR

                        Then

                            # for parallel FFT, VASP <= 5.2.x
                            wfc = Wavecar('WAVECAR', gamma=True, gamma_half='z')

                            # for serial FFT, VASP <= 5.2.x
                            wfc = Wavecar('WAVECAR', gamma=True, gamma_half='x')

                        For VASP >= 5.4, WAVECAR is written with x-direction half grid regardless of
                        parallel or serial FFT.

                            # "gamma_half" default to "x" for VASP >= 5.4
                            wfc = Wavecar('WAVECAR', gamma=True, gamma_half='x')
                        ''')
                else:
                    raise ValueError('''
                    NUMBERS OF PLANEWAVES NOT CONSISTENT:

                        THIS CODE -> %d
                        FROM VASP -> %d
                           NGRIDS -> %d
                    ''' % (Gvector.shape[0],
                           self.nplanewaves[kpoint_index - 1] // 2 if self.soc else self.nplanewaves[kpoint_index - 1],
                           int(np.prod(self.planewave_grid))))

        return np.asarray(Gvector, dtype=int)

    def wfc_r(self, spin_index=1, kpoint_index=1, band_index=1,
              gvector=None, coef=None, fft_grid=None, rescale=None, norm=False, kr_phase=False,
              r_shift=[0.0, 0.0, 0.0]):
        """
        Obtain the pseudo-wavefunction of the specified KS states in real space
        by performing FT transform on the reciprocal space planewave coefficients.
        The 3D FT grid size is determined by fft_grid, which defaults to self.planewave_grid if not given.
        Gvectors of the KS states is used to put 1D planewave coefficients back to 3D grid.

        Inputs:
            spin_index    : spin index of the desired KS states, starting from 1
            kpoint_index  : k-point index of the desired KS states, starting from 1
            band_index    : band index of the desired KS states, starting from 1
            gvector       : the G-vectors correspond to the plane-wave coefficients
            coef          : the plane-wave coefficients. If None, read from WAVECAR
            fft_grid : the FFT grid size
            norm  : whether normalized coef ?
         kr_phase : whether multiply the exp(ikr) phase ?
               r0 : shift of the kr-phase to get full wfc other than primitive cell

        The return wavefunctions are normalized in a way that

                        \sum_{ijk} | \phi_{ijk} | ^ 2 = 1

        """
        self.checkIndex(spin_index, kpoint_index, band_index)

        if fft_grid is None:
            fft_grid = self.fft_grid
        else:
            fft_grid = np.array(fft_grid, dtype=int)
            assert fft_grid.shape == (3,)
            assert np.alltrue(fft_grid >= self.planewave_grid), f"Minium FT grid size: " \
                                                                f"({self.planewave_grid})"
        # logger.debug(f"plane-wave grid {self.planewave_grid}\n"
        #              f"fft grid {fft_grid}\n")

        # By default, the WAVECAR only stores the periodic part of the Bloch wavefunction.
        # In order to get the full Bloch wavefunction, one need to
        # multiply the periodic part with the phase: exp(i k (r + r0)).
        # Below, the k-point vector and the real-space grid are both in the direct coordinates.
        if kr_phase:
            r = np.mgrid[0:fft_grid[0], 0:fft_grid[1], 0:fft_grid[2]].reshape(
                (3, np.prod(fft_grid))).T / fft_grid.astype(float)
            r0 = np.array(r_shift, dtype=float)
            phase = np.exp(1j * np.pi * 2 * np.sum(self.kpoint_vectors[kpoint_index - 1] * (r + r0), axis=1)).reshape(
                fft_grid)
        else:
            phase = 1.0

        # The default normalization of np.fft.fftn has the direct transforms
        # unscaled and the inverse transforms are scaled by 1/n. It is possible
        # to obtain unitary transforms by setting the keyword argument norm to
        # "ortho" (default is None) so that both direct and inverse transforms
        # will be scaled by 1/\sqrt{n}.

        # default normalization factor so that
        # \sum_{ijk} | \phi_{ijk} | ^ 2 = 1
        normalization_factor = rescale if rescale is not None else np.sqrt(np.prod(fft_grid))
        # logger.debug(f"normalized factor {normalization_factor}")

        if gvector is None:
            gvector = self.gvectors(kpoint_index)

        if self.gamma:
            if self.gamma_half == 'x':
                phi_k = np.zeros((fft_grid[0] // 2 + 1, fft_grid[1], fft_grid[2]), dtype=np.complex128)
            else:
                phi_k = np.zeros((fft_grid[0], fft_grid[1], fft_grid[2] // 2 + 1), dtype=np.complex128)
        else:
            phi_k = np.zeros(fft_grid, dtype=np.complex128)

        gvector %= fft_grid[np.newaxis, :]
        dump = coef if coef else self.read_coef(spin_index, kpoint_index, band_index, norm)
        assert dump.shape[0] == gvector.shape[0], "the shape of gvector should match the plane_wave coefficients"

        if self.soc:
            wfc_spinor = []
            nplane_waves = dump.shape[0] // 2
            # spinor up
            phi_k[gvector[:, 0], gvector[:, 1], gvector[:, 2]] = dump[:nplane_waves]
            wfc_spinor.append(np.fft.ifftn(phi_k) * normalization_factor * phase)
            # spinor down
            # TODO(lmz):proof it is not necessary
            phi_k[:, :, :] = 0.0j
            phi_k[gvector[:, 0], gvector[:, 1], gvector[:, 2]] = dump[nplane_waves:]
            wfc_spinor.append(np.fft.ifftn(phi_k) * normalization_factor * phase)
            del dump
            return wfc_spinor

        else:
            phi_k[gvector[:, 0], gvector[:, 1], gvector[:, 2]] = dump
            if self.gamma:
                # add some components that are excluded and perform complex to real FFT
                if self.gamma_half == 'z':
                    for grid_index0 in range(fft_grid[0]):
                        for grid_index1 in range(fft_grid[1]):
                            fx = grid_index0 if grid_index0 < fft_grid[0] // 2 + 1 else grid_index0 - fft_grid[0]
                            fy = grid_index1 if grid_index1 < fft_grid[1] // 2 + 1 else grid_index1 - fft_grid[1]
                            if (fy > 0) or (fy == 0 and fx >= 0):
                                continue
                            phi_k[grid_index0, grid_index1, 0] = phi_k[-grid_index0, -grid_index1, 0].conjugate()

                    # VASP add a factor of SQRT2 for G != 0 in Gamma-only VASP
                    phi_k /= np.sqrt(2.)
                    phi_k[0, 0, 0] *= np.sqrt(2.)
                    return np.fft.irfftn(phi_k, s=fft_grid) * normalization_factor
                elif self.gamma_half == 'x':
                    for grid_index1 in range(fft_grid[1]):
                        for grid_index2 in range(fft_grid[2]):
                            fy = grid_index1 if grid_index1 < fft_grid[1] // 2 + 1 else grid_index1 - fft_grid[1]
                            fz = grid_index2 if grid_index2 < fft_grid[2] // 2 + 1 else grid_index2 - fft_grid[2]
                            if (fy > 0) or (fy == 0 and fz >= 0):
                                continue
                            phi_k[0, grid_index1, grid_index2] = phi_k[0, -grid_index1, -grid_index2].conjugate()

                    phi_k /= np.sqrt(2.)
                    phi_k[0, 0, 0] *= np.sqrt(2.)
                    phi_k = np.swapaxes(phi_k, 0, 2)
                    tmp = np.fft.irfftn(
                        phi_k, s=(fft_grid[2], fft_grid[1], fft_grid[0])) * normalization_factor
                    return np.swapaxes(tmp, 0, 2)
            else:
                # TODO(lmz): std test
                # perform complex2complex FFT
                return np.fft.ifftn(phi_k) * normalization_factor * phase

    def save2vesta(self, phi=None, lreal=False, poscar='POSCAR', prefix='wfc', ncol=10):
        """
        Save the real space pseudo-wavefunction as vesta format.
        """
        # TODO(lmz): deal with spin down (spin up currently)
        if self.soc == True:
            psi = phi[0].copy()
        else:
            psi = phi.copy()

        nx, ny, nz = psi.shape
        try:
            pos = open(poscar, 'r')
            head = ''
            for line in pos:
                if line.strip():
                    head += line
                else:
                    break
            head += f'\n{nx:5d}{ny:5d}{nz:5d}\n'
        except:
            raise IOError('Failed to open %s' % poscar)

        # Faster IO
        nrow = psi.size // ncol
        nremnant = psi.size % ncol

        psi = psi.flatten(order='F')
        psi_main = psi[:nrow * ncol].reshape((nrow, ncol))
        psi_remnant = psi[nrow * ncol:]

        with open(prefix + '_r.vasp', 'w') as out:
            out.write(head)
            out.write('\n'.join([''.join([f"{i:16.8E}" for i in row]) for row in psi_main.real]))
            out.write("\n" + ''.join([f"{i:16.8E}" for i in psi_remnant.real]))
        # the wavefunctions in real-space can be made to have zero imaginary components -- i.e. entirely real
        if not (self.gamma or lreal):
            with open(prefix + '_i.vasp', 'w') as out:
                out.write(head)
                out.write('\n'.join([''.join([f"{i:16.8E}" for i in row]) for row in psi_main.imag]))
                out.write("\n" + ''.join([f"{i:16.8E}" for i in psi_remnant.imag]))

    def poisson(self, rho=None, band_index=1, kpoint_index=1, spin_index=1, fft_grid=None, norm=False):
        """
        Given a charge density "rho", solve the Poisson equation with periodic
        boundary condition to find out the corresponding electric potential and
        field.

        When "rho" is None, construct the charge density from a chosen Kohn-Sham
        state, i.e. rho(r) = phi_n(r).conj() * phi_n(r).

        In SI units, the real space Poisson equation:

                    \nabla^2 V = - \rho / \varepsilon_0
                             E = - \nabla V

        the reciprocal space Poisson equation:

                    G**2 * V_q = - rho_q / \varepsilon_0
                           E_q = -1j * G * V_q

        Note that the G=(0,0,0) entry is set to 1.0 instead of 0 to avoid
        divergence.
        """

        if rho is not None:
            rho = np.asarray(rho)
            fft_grid = np.array(rho.shape, dtype=int)
            assert fft_grid.shape == (3,)
        else:
            fft_grid = self.fft_grid
            # normalization factor so that
            # \sum_{ijk} | \phi_{ijk} | ^ 2 * volume / Ngrid = 1
            normFac = np.prod(fft_grid) / self.omega
            if self.soc:
                rho = np.zeros(fft_grid, dtype=float)
                phi_spinor = self.wfc_r(spin_index=spin_index, kpoint_index=kpoint_index, band_index=band_index,
                                        fft_grid=fft_grid, norm=norm)
                # negative charges, hence the minus sign
                for phi in phi_spinor:
                    rho += -(phi.conj() * phi).real * normFac
            else:
                phi = self.wfc_r(spin_index=spin_index, kpoint_index=kpoint_index, band_index=band_index,
                                 fft_grid=fft_grid, norm=norm)
                # negative charges, hence the minus sign
                rho = -(phi.conj() * phi).real * normFac

        fx = [ii if ii < fft_grid[0] // 2 + 1 else ii - fft_grid[0]
              for ii in range(fft_grid[0])]
        fy = [jj if jj < fft_grid[1] // 2 + 1 else jj - fft_grid[1]
              for jj in range(fft_grid[1])]
        fz = [kk if kk < fft_grid[2] // 2 + 1 else kk - fft_grid[2]
              for kk in range(fft_grid[2])]

        # plane-waves: Reciprocal coordinate
        # indexing = 'ij' so that outputs are of shape (fft_grid[0], fft_grid[1], fft_grid[2])
        Dx, Dy, Dz = np.meshgrid(fx, fy, fz, indexing='ij')
        # plane-waves: Cartesian coordinate
        Gx, Gy, Gz = np.tensordot(
            self.reciprocal * np.pi * 2, [Dx, Dy, Dz], axes=(0, 0))
        # the norm squared of the G-vectors
        G2 = Gx ** 2 + Gy ** 2 + Gz ** 2
        # Note that the G=(0,0,0) entry is set to 1.0 instead of 0.
        G2[0, 0, 0] = 1.0

        # permittivity of vacuum [F / m]
        _eps0 = 8.85418781762039E-12
        # charge of one electron, in unit of Coulomb [1F * 1V]
        _e = 1.6021766208E-19

        # charge density in reciprocal space, rho in unit of [Coulomb / Angstrom**3]
        rho_q = np.fft.fftn(1E10 * _e * rho / _eps0, norm='ortho')
        # the electric potential in reciprocal space
        # V_q = -rho_q / (-G2)
        V_q = rho_q / G2
        # the electric potential in real space in unit of 'Volt'
        V_r = np.fft.ifftn(V_q, norm='ortho').real
        # the electric field in x/y/z in real space in unit of 'Volt / Angstrom'
        E_x = np.fft.ifftn(-1j * Gx * V_q, norm='ortho').real
        E_y = np.fft.ifftn(-1j * Gy * V_q, norm='ortho').real
        E_z = np.fft.ifftn(-1j * Gz * V_q, norm='ortho').real

        return rho, V_r, E_x, E_y, E_z

    def TransitionDipoleMoment(self, ks_i, ks_j):
        """
        """
        return self.get_dipole_mat(ks_i, ks_j)

    def get_dipole_mat(self, ks_i, ks_j):
        """
        Dipole transition within the electric dipole approximation (EDA).
        Please refer to this post for more details.

          https://qijingzheng.github.io/posts/Light-Matter-Interaction-and-Dipole-Transition-Matrix/

        The dipole transition matrix elements in the length gauge is given by:

                <psi_nk | e r | psi_mk>

        where | psi_nk > is the pseudo-wavefunction.  In periodic systems, the
        position operator "r" is not well-defined.  Therefore, we first evaluate
        the momentum operator matrix in the velocity gauge, i.e.

                <psi_nk | p | psi_mk>

        And then use simple "p-r" relation to apprimate the dipole transition
        matrix element

                                          -i⋅h
            <psi_nk | r | psi_mk> =  -------------- ⋅ <psi_nk | p | psi_mk>
                                       m⋅(En - Em)

        Apparently, the above equaiton is not valid for the case Em == En. In
        this case, we just set the dipole matrix element to be 0.

        ################################################################################
        NOTE that, the simple "p-r" relation only applies to molecular or finite
        system, and there might be problem in directly using it for periodic
        system. Please refer to this paper for more details.

          "Relation between the interband dipole and momentum matrix elements in
          semiconductors"
          (https://journals.aps.org/prb/pdf/10.1103/PhysRevB.87.125301)

        ################################################################################
        """

        # ks_i and ks_j are list containing spin-, kpoint- and band-index of the
        # initial and final states
        assert len(ks_i) == len(ks_j) == 3, 'Must be three indexes!'
        assert ks_i[1] == ks_j[1], 'k-point of the two states differ!'
        self.checkIndex(*ks_i)
        self.checkIndex(*ks_j)

        # energy differences between the two states
        Emk = self.bands[ks_i[0] - 1, ks_i[1] - 1, ks_i[2] - 1]
        Enk = self.bands[ks_j[0] - 1, ks_j[1] - 1, ks_j[2] - 1]
        dE = Enk - Emk

        # if energies of the initial and final states are the same, set the
        # dipole transition moment zero.
        if np.allclose(dE, 0.0):
            return 0.0

        moment_mat = self.get_moment_mat(ks_i, ks_j)
        dipole_mat = -1j / (dE / (2 * Ry2eV)) * moment_mat * au2angstrom * au2Debye

        return Emk, Enk, dE, dipole_mat

    def get_moment_mat(self, ks_i, ks_j):
        """
        The momentum operator matrix between the pseudo-wavefunction in the
        velocity gauge

            <psi_nk | p | psi_mk> = hbar <u_nk | k - i nabla | u_mk>

        The nabla operator matrix elements between the pseudo-wavefuncitons

            <u_nk | k - i nabla | u_mk>

           = \sum_G C_nk(G).conj() * C_mk(G) * [k + G]

        where C_nk(G) is the plane-wave coefficients for | u_nk >.
        """

        # ks_i and ks_j are list containing spin-, kpoint- and band-index of the
        # initial and final states
        assert len(ks_i) == len(ks_j) == 3, 'Must be three indexes!'
        assert ks_i[1] == ks_j[1], 'k-point of the two states differ!'
        self.checkIndex(*ks_i)
        self.checkIndex(*ks_j)

        # k-points in direct coordinate
        k0 = self.kpoint_vectors[ks_i[1] - 1]
        # plane-waves in direct coordinates
        G0 = self.gvectors(kpoint_index=ks_i[1])
        # G + k in Cartesian coordinates
        Gk = np.dot(
            G0 + k0,  # G in direct coordinates
            self.reciprocal * 2 * np.pi  # reciprocal basis x 2pi
        )

        # plane-wave coefficients for initial (mk) and final (nk) states
        CG_mk = self.read_coef()
        CG_nk = self.read_coef()
        ovlap = CG_nk.conj() * CG_mk

        ################################################################################
        # Momentum operator matrix element between pseudo-wavefunctions
        ################################################################################
        if self.gamma:
            # for gamma-only, only half the plane-wave coefficients are stored.
            # Moreover, the coefficients are multiplied by a factor of sqrt2

            # G > 0 part
            moment_mat_ps = np.sum(ovlap[:, None] * Gk, axis=0)

            # For gamma-only version, add the other half plane-waves, G_ = -G
            # G < 0 part, C(G) = C(-G).conj()
            moment_mat_ps -= np.sum(
                ovlap[:, None].conj() * Gk,
                axis=0)

            # remove the sqrt2 factor added by VASP
            moment_mat_ps /= 2.0

        elif self.soc:
            moment_mat_ps = np.sum(
                ovlap[:, None] * np.r_[Gk, Gk],
                axis=0)
        else:
            moment_mat_ps = np.sum(
                ovlap[:, None] * Gk, axis=0
            )

        return moment_mat_ps

    # def inverse_participation_ratio(self, norm=True):
    #     '''
    #     Calculate Inverse Paticipation Ratio (IPR) from the wavefunction. IPR is
    #     a measure of the localization of Kohn-Sham states. For a particular KS
    #     state \phi_j, it is defined as
    #
    #                         \sum_n |\phi_j(n)|^4
    #         IPR(\phi_j) = -------------------------
    #                       |\sum_n |\phi_j(n)|^2||^2
    #
    #     where n iters over the number of grid points.
    #     '''
    #
    #     self.ipr = np.zeros((self.nspin, self.nkpoints, self.nbands, 3))
    #
    #     for spin_index in range(self.nspin):
    #         for kpoint_index in range(self.nkpoints):
    #             for band_index in range(self.nbands):
    #                 phi_j = self.wfc_r(spin_index + 1, kpoint_index + 1, band_index + 1, norm=norm)
    #                 phi_j_abs = np.abs(phi_j)
    #
    #                 print('Calculating IPR of #spin %4d, #kpoint %4d, #band %4d' %
    #                       (spin_index + 1, kpoint_index + 1, band_index + 1))
    #                 self.ipr[spin_index, kpoint_index, band_index,
    #                          0] = self.kpath[kpoint_index] if self.kpath is not None else 0
    #                 self.ipr[spin_index, kpoint_index, band_index,
    #                          1] = self.bands[spin_index, kpoint_index, band_index]
    #                 self.ipr[spin_index, kpoint_index, band_index, 2] = np.sum(
    #                     phi_j_abs ** 4) / np.sum(phi_j_abs ** 2) ** 2
    #
    #     np.save('ipr.npy', self.ipr)
    #     return self.ipr

    def elf(self, kpoint_weight, fft_grid=None, warn=True):
        '''
        Calculate the electron localization function (ELF) from WAVECAR.

        The following formula was extracted from VASP ELF.F:
                     _
                     h^2    *    2      T.........kinetic energy
          T    =  -2 --- Psi grad Psi   T+TCORR...pos.definite kinetic energy
                   ^ 2 m                TBOS......T of an ideal Bose-gas
                   ^
                   I am not sure if we need to time 2 here, use 1 in this
                   script.

                   _                                (=infimum of T+TCORR)
                 1 h^2      2           DH........T of hom.non-interact.e- - gas
          TCORR= - ---  grad rho                    (acc.to Fermi)
                 2 2 m                  ELF.......electron-localization-function
                   _             2
                 1 h^2 |grad rho|
          TBOS = - --- ----------       D = T + TCORR - TBOS
                 4 2 m    rho
                   _                                \                1
                 3 h^2        2/3  5/3          =====>    ELF = ------------
          DH   = - --- (3 Pi^2)  rho                /                   D   2
                 5 2 m                                           1 + ( ---- )
                                                                        DH

        REF:
            1. Nature, 371, 683-686 (1994)
            2. Becke and Edgecombe, J. Chem. Phys., 92, 5397(1990)
            3. M. Kohout and A. Savin, Int. J. Quantum Chem., 60, 875-882(1996)
            4. http://www2.cpfs.mpg.de/ELF/index.php?content=06interpr.txt
        '''

        if warn:
            warning = """
            ###################################################################
            If you are using VESTA to view the resulting ELF, please rename the
            output file as ELFCAR, otherwise there will be some error in the
            isosurface plot!

            When CHG*/PARCHG/*.vasp are read in to visualize isosurfaces and
            sections, data values are divided by volume in the unit of bohr^3.
            The unit of charge densities input by VESTA is, therefore, bohr^−3.

            For LOCPOT/ELFCAR files, volume data are kept intact.

            You can turn off this warning by setting "warn=False" in the "elf"
            method.
            ###################################################################
            """
            print(warning)

        # the k-point weights
        kpoint_weight = np.array(kpoint_weight, dtype=float)
        assert kpoint_weight.shape == (self.nkpoints,), "K-point weights must be provided to calculate charge density!"
        # normalization
        kpoint_weight /= kpoint_weight.sum()

        if fft_grid is None:
            fft_grid = self.fft_grid
        else:
            fft_grid = np.array(fft_grid, dtype=int)
            assert fft_grid.shape == (3,)
            assert np.alltrue(fft_grid >= self.planewave_grid), f"Minium FT grid size: {self.planewave_grid}"

        fx = [i if i < fft_grid[0] // 2 + 1 else i - fft_grid[0] for i in range(fft_grid[0])]
        fy = [i if i < fft_grid[1] // 2 + 1 else i - fft_grid[1] for i in range(fft_grid[1])]
        fz = [i if i < fft_grid[2] // 2 + 1 else i - fft_grid[2] for i in range(fft_grid[2])]

        # plane-waves: Reciprocal coordinate
        # indexing = 'ij' so that outputs are of shape (fft_grid[0], fft_grid[1], fft_grid[2])
        Dx, Dy, Dz = np.meshgrid(fx, fy, fz, indexing='ij')
        # plane-waves: Cartesian coordinate
        Gx, Gy, Gz = np.tensordot(self.reciprocal * np.pi * 2, [Dx, Dy, Dz], axes=(0, 0))
        # the norm squared of the G-vectors
        G2 = Gx ** 2 + Gy ** 2 + Gz ** 2
        # k-points vectors in Cartesian coordinate
        kpoint_vectors_Cartesian = np.dot(self.kpoint_vectors, self.reciprocal * 2 * np.pi)

        # normalization factor so that \sum_{ijk} | \phi_{ijk} | ^ 2 * volume / Ngrid = 1
        normalization_factor = np.sqrt(np.prod(fft_grid) / self.omega)

        # electron localization function
        ElectronLocalizationFunction = []
        # Charge density
        rho = np.zeros(fft_grid, dtype=complex)
        # Kinetic energy density
        tau = np.zeros(fft_grid, dtype=complex)

        for spin_index in range(self.nspin):
            # initialization
            rho[...] = 0.0
            tau[...] = 0.0

            for kpoint_index in range(self.nkpoints):

                # plane-wave G-vectors
                gvectors_for_kpoint = self.gvectors(kpoint_index + 1)
                # for gamma-only version, restore the missing -G vectors
                if self.gamma:
                    tmp = np.array([-k for k in gvectors_for_kpoint[1:]], dtype=int)
                    gvectors_for_kpoint = np.vstack([gvectors_for_kpoint, tmp])
                # plane-wave G-vectors in Cartesian coordinate
                gvectors_for_kpoint_Cartesian = np.dot(gvectors_for_kpoint, self.reciprocal * 2 * np.pi)

                k = kpoint_vectors_Cartesian[kpoint_index]
                # G + k
                g_plus_k = gvectors_for_kpoint_Cartesian + k[np.newaxis, :]
                # | G + k |^2
                g_plus_k_squared = np.linalg.norm(g_plus_k, axis=1) ** 2

                for band_index in range(self.nbands):
                    # omit the empty bands
                    if self.occupations[spin_index, kpoint_index, band_index] == 0.0:
                        continue

                    electron_degeneracy = 2.0 if self.nspin == 1 else 1.0
                    weight = electron_degeneracy * kpoint_weight[kpoint_index] * \
                             self.occupations[spin_index, kpoint_index, band_index]

                    # wavefunction in reciprocal space
                    # VASP does NOT do normalization in elf.F
                    phi_q = self.read_coef(spin_index=spin_index + 1, kpoint_index=kpoint_index + 1,
                                           band_index=band_index + 1,
                                           norm=False)
                    # pad the missing planewave coefficients for -G vectors
                    if self.gamma:
                        tmp = [x.conj() for x in phi_q[1:]]
                        phi_q = np.concatenate([phi_q, tmp])
                        # Gamma only, divide a factor of sqrt(2.0) except for G=0
                        phi_q /= np.sqrt(2.0)
                        phi_q[0] *= np.sqrt(2.0)
                    # wavefunction in real space
                    phi_r = self.wfc_r(spin_index=spin_index + 1, kpoint_index=kpoint_index + 1,
                                       band_index=band_index + 1, gvector=gvectors_for_kpoint, coef=phi_q,
                                       fft_grid=fft_grid) * normalization_factor
                    # grad^2 \phi in reciprocal space
                    lap_phi_q = -g_plus_k_squared * phi_q
                    # grad^2 \phi in real space
                    lap_phi_r = self.wfc_r(spin_index=spin_index + 1, kpoint_index=kpoint_index + 1,
                                           band_index=band_index + 1, gvector=gvectors_for_kpoint, coef=lap_phi_q,
                                           fft_grid=fft_grid) * normalization_factor

                    # \phi* grad^2 \phi in real space --> kinetic energy density
                    tau += -phi_r * lap_phi_r.conj() * weight

                    # charge density in real space
                    rho += phi_r.conj() * phi_r * weight

            # charge density in reciprocal space
            rho_q = np.fft.fftn(rho, norm='ortho')

            # grad^2 rho: laplacian of charge density
            lap_rho_q = -G2 * rho_q
            lap_rho_r = np.fft.ifftn(lap_rho_q, norm='ortho')

            # charge density gradient: grad rho
            ########################################
            # wrong method for gradient using FFT
            ########################################
            # grad_rho_x = np.fft.ifft(1j * Gx * np.fft.fft(rho, axis=0), axis=0)
            # grad_rho_y = np.fft.ifft(1j * Gy * np.fft.fft(rho, axis=1), axis=1)
            # grad_rho_z = np.fft.ifft(1j * Gz * np.fft.fft(rho, axis=2), axis=2)

            ########################################
            # correct method for gradient using FFT
            ########################################
            grad_rho_x = np.fft.ifftn(1j * Gx * rho_q, norm='ortho')
            grad_rho_y = np.fft.ifftn(1j * Gy * rho_q, norm='ortho')
            grad_rho_z = np.fft.ifftn(1j * Gz * rho_q, norm='ortho')

            grad_rho_sq = np.abs(grad_rho_x) ** 2 \
                          + np.abs(grad_rho_y) ** 2 \
                          + np.abs(grad_rho_z) ** 2

            rho = rho.real
            tau = tau.real
            lap_rho_r = lap_rho_r.real

            Cf = 3. / 5 * (3.0 * np.pi ** 2) ** (2. / 3)
            Dh = np.where(rho > 0.0,
                          Cf * rho ** (5. / 3),
                          0.0)
            eps = 1E-8 / HSQDTM
            Dh[Dh < eps] = eps
            # D0 = T + TCORR - TBOS
            D0 = tau + 0.5 * lap_rho_r - 0.25 * grad_rho_sq / rho

            ElectronLocalizationFunction.append(1. / (1. + (D0 / Dh) ** 2))

        return ElectronLocalizationFunction


if __name__ == '__main__':
    wf = Wavecar('WAVECAR', soc=False, gamma=True, gamma_half='x', grid_scale=2)
    for i in [637, 638, 639]:
        phi = wf.wfc_r(spin_index=1, kpoint_index=1, band_index=431, kr_phase=False)
        wf.save2vesta(phi=np.abs(phi) ** 2, lreal=True, poscar='POSCAR', prefix=str(i), ncol=10)

    # wf = Wavecar('WAVECAR_gamma', soc=False, gamma=True, gamma_half='x', grid_scale=1)
    # phi = wf.wfc_r(spin_index=1, kpoint_index=1, band_index=431, kr_phase=False)
    # wf.save2vesta(phi, poscar='POSCAR_gamma')

    # wf = Wavecar('WAVECAR_std', soc=False, gamma=False, gamma_half='x', grid_scale=1)
    # print(wf.__dict__)
    # phi = wf.wfc_r(spin_index=1, kpoint_index=1, band_index=431, kr_phase=True)
    # wf.save2vesta(phi, poscar='POSCAR_std', prefix="wfc_1")

    # wf = Wavecar('WAVECAR_soc', soc=True, gamma=False, gamma_half='x', grid_scale=1)
    # phi = wf.wfc_r(kpoint_index=1, band_index=27, kr_phase=True)
    # wf.save2vesta(phi, poscar='POSCAR_soc')

    # xx = Wavecar('wavecar')
    # phi = xx.wfc_r(1, 30, 17, fft_grid=(28, 28, 252))
    # xx.save2vesta(phi, poscar='POSCAR')

    # xx = Wavecar('./gamma/WAVECAR')
    # phi = xx.wfc_r(1, 1, 317, fft_grid=(60, 108, 160),
    #                gamma=True)
    # xx.save2vesta(phi, poscar='./gamma/POSCAR',gamma=True)

    # xx = Wavecar('WAVECAR')
    # dE, ovlap, tdm = xx.TransitionDipoleMoment([1,30,17], [1,30,18], norm=True)
    # print dE, ovlap.real, np.abs(tdm)**2
    # print xx._kpath
    # b = xx.readBandCoeff(1,1,1)
    # xx = np.savetxt('kaka.dat', xx.gvectors(2), fmt='%5d')
    # gvec = xx.gvectors(1)
    # gvec %= xx.planewave_grid[np.newaxis, :]
    # print gvec

    # fft_grid=(28, 28, 252)
    # phi = xx.wfc_r(1, 30, 17, fft_grid=(28, 28, 252))
    # header = open('POSCAR').read()
    # with open('wave_real.vasp', 'w') as out:
    #     out.write(header)
    #     out.write('%5d%5d%5d\n' % (fft_grid[0], fft_grid[1], fft_grid[2]))
    #     nwrite=0
    #     for kk in range(fft_grid[2]):
    #         for jj in range(fft_grid[1]):
    #             for ii in range(fft_grid[0]):
    #                 nwrite += 1
    #                 out.write('%22.16f ' % phi.real[ii,jj,kk])
    #                 if nwrite % 10 == 0:
    #                     out.write('\n')
    # with open('wave_imag.vasp', 'w') as out:
    #     out.write(header)
    #     out.write('%5d%5d%5d\n' % (fft_grid[0], fft_grid[1], fft_grid[2]))
    #     nwrite=0
    #     for kk in range(fft_grid[2]):
    #         for jj in range(fft_grid[1]):
    #             for ii in range(fft_grid[0]):
    #                 nwrite += 1
    #                 out.write('%22.16f ' % phi.imag[ii,jj,kk])
    #                 if nwrite % 10 == 0:
    #                     out.write('\n')

    # xx = Wavecar('wave_tyz')
    # ipr = xx.inverse_participation_ratio()
    # print xx.nbands, xx.nkpoints
    #
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = plt.subplot()
    #
    # ax.scatter(ipr[...,0], ipr[..., 1], s=ipr[..., 2] / ipr[..., 2].max() * 10, c=ipr[..., 2],
    #            cmap='jet_r')
    #
    # plt.show()

    # wfc = Wavecar('WAVECAR', gamma=True, gamma_half='x')
    # # fft_grid = [80, 140, 210]
    # phi = wfc.wfc_r(band_index=190)
    #
    # rho = np.abs(phi) ** 2
    # # rho2 = VaspChargeDensity('PARCHG.0158.ALLK').chg[0]
    # # rho /= rho.sum()
    # # rho2 /= rho2.sum()
    # # rho3 = rho - rho2
    #
    # wfc.save2vesta(rho, lreal=True)
