#!/usr/bin/env python3

import numpy as np
import os

""" Interpolation from real-space to k-space of
(k-derivatives of) matrix elements employing FFTs.
"""

class KspaceInterpolator:

    def __init__(self, lattice, RcellShift, **rMats):
        self.lattice = lattice
        self.minIndices = RcellShift
        self.rMats = rMats

    def to_frac_k(self, k_au):
        return self.lattice @ k_au / (2 * np.pi)

    def from_frac_k(self, kf):
        return 2*np.pi * np.linalg.inv(self.lattice) @ kf

    def enforce_hermiticity(self, key=None):
        if key is None:
            for k in self.rMats.keys():
                self.enforce_hermiticity(k)
            return
        mi = self.minIndices
        M = self.rMats[key]
        Mnew = np.zeros(M.shape, dtype=complex)
        for x in range(M.shape[0]):
            for y in range(M.shape[1]):
                for z in range(M.shape[2]):
                    Ro = (x, y, z)
                    Rm = (-2*mi[0]-x, -2*mi[1]-y, -2*mi[2]-z)
                    Mnew[*Ro] = 0.5 * (M[*Ro] + M[*Rm].swapaxes(0, 1).conj())
        self.rMats[key] = Mnew

    def restrict_to_real_or_imag(self, key=None, thres=1e-3):
        if key is None:
            for k in self.rMats.keys():
                self.restrict_to_real_or_imag(k)
            return
        M0 = self.rMats[key][-self.minIndices]
        reMax = np.max(np.abs(np.real(M0)))
        imMax = np.max(np.abs(np.imag(M0)))
        if reMax * thres > imMax:
            self.rMats[key] = np.real(self.rMats[key])
        elif imMax * thres > reMax:
            self.rMats[key] = 1j * np.imag(self.rMats[key])
        else:
            print(f"'{key}' is neither purely real nor purely imaginary")

    def expand_grid(self, Nw=None, key=None):
        if isinstance(Nw, int):
            Nw = (Nw, Nw, Nw)
        assert len(Nw) == 3
        if key is None:
            for k in self.rMats.keys():
                self.expand_grid(Nw, k)
            return
        M = self.rMats[key]
        assert Nw[0] >= M.shape[0]
        assert Nw[1] >= M.shape[1]
        assert Nw[2] >= M.shape[2]
        newM = np.zeros((*Nw, *M.shape[3::]), dtype=complex)
        newM[0:M.shape[0], 0:M.shape[1], 0:M.shape[2]] = M
        self.rMats[key] = newM


    def _k_grid_single(self, M, kFrac):
        sp = M.shape
        gx, gy, gz = np.mgrid[0:sp[0], 0:sp[1], 0:sp[2]]
        phase = np.exp(2j * np.pi * np.modf(gx * kFrac[0] + gy * kFrac[1] + gz * kFrac[2])[0])
        l = tuple([None] * (len(sp)-3))
        T = M * phase[:, :, :, *l]
        T = np.fft.ifftn(T, axes=(0, 1, 2), norm="forward")
        phase = np.exp(2j * np.pi * np.modf(np.dot(kFrac, self.minIndices) +
                                     gx * self.minIndices[0] / sp[0] +
                                     gy * self.minIndices[1] / sp[1] +
                                     gz * self.minIndices[2] / sp[2] )[0]
                      )
        T *= phase[:, :, :, *l]
        return T

    def _k_grid_repeated(self, M, kFrac, repeat=(2, 2, 2)):
        # can be implemented more efficient...
        T = np.empty((M.shape[0]*repeat[0], M.shape[1]*repeat[1], M.shape[2]*repeat[2], *M.shape[3::]), dtype=complex)
        offBare = np.array([ 1/(M.shape[0]*repeat[0]), 1/(M.shape[1]*repeat[1]), 1/(M.shape[2]*repeat[2])])
        for a in range(repeat[0]):
            for b in range(repeat[1]):
                for c in range(repeat[2]):
                    T[a::repeat[0], b::repeat[1], c::repeat[2]] = self._k_grid_single(M, kFrac +
                                                                       np.array([offBare[0]*a, offBare[1]*b, offBare[2]*c]))
        return T

    def _k_grid_mat(self, M, kFrac, repeat):
        if repeat is None:
            return self._k_grid_single(M, kFrac)
        if isinstance(repeat, int):
            return self._k_grid_repeated(M, kFrac, (repeat, repeat, repeat))
        assert len(repeat) == 3
        return self._k_grid_repeated(M, kFrac, repeat)


    def k_grid(self, key=None, kFrac=np.array([0, 0, 0]), repeat=None):
        if key is None:
            result = {}
            for k in self.rMats.keys():
                result[k] = self.k_grid(k, kFrac, repeat)
            return result
        return self._k_grid_mat(self.rMats[key], kFrac, repeat)


    def dk_grid(self, key=None, repeat=None, kFrac=np.array([0, 0, 0])):
        if key is None:
            result = {}
            for k in self.rMats.keys():
                result[k] = self.dk_grid(k, repeat, kFrac)
            return result
        M = self.rMats[key]
        sp = M.shape
        mi = self.minIndices
        g = np.mgrid[mi[0]:sp[0]+mi[0], mi[1]:sp[1]+mi[1], mi[2]:sp[2]+mi[2]]
        pos = np.einsum("vxyz,vd->xyzd", g, self.lattice)
        irM = np.einsum("xyz...,xyzd->xyz...d", M, 1j * pos)
        return self._k_grid_mat(irM, kFrac, repeat)

    def dk_points(self, key, kFracs):
        if len(kFracs.shape) == 1:
            kFracs = np.array([kFracs])
        M = self.rMats[key]
        sp = M.shape
        mi = self.minIndices
        g = np.mgrid[mi[0]:sp[0]+mi[0], mi[1]:sp[1]+mi[1], mi[2]:sp[2]+mi[2]]
        pos = np.einsum("vxyz,vd->xyzd", g, self.lattice)
        irM = np.einsum("xyz...,xyzd->xyz...d", M, 1j * pos)
        ikr = 2j * np.pi * np.einsum("dabc,sd->sabc", g, kFracs)
        return np.einsum("abc...,sabc->s...", irM, np.exp(ikr))

    def k_points(self, key, kFracs):
        if len(kFracs.shape) == 1:
            kFracs = np.array([kFracs])
        M = self.rMats[key]
        sp = M.shape
        mc = self.minIndices
        g = np.mgrid[mc[0]:sp[0]+mc[0], mc[1]:sp[1]+mc[1], mc[2]:sp[2]+mc[2]]
        ikr = 2j * np.pi * np.einsum("dabc,sd->sabc", g, kFracs)
        return np.einsum("abc...,sabc->s...", M, np.exp(ikr))
