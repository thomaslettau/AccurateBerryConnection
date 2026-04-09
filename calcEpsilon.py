#!/usr/bin/env python3

""" Computes the (linear) complex dielectric constant based on epsilon.x (Drude model) for all available velocity operators
"""

import argparse
from datetime import timedelta
import functools
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import psutil
import sys
import time
from threadpoolctl import threadpool_limits

import atu
from KspaceInterpolator import KspaceInterpolator


def extractFermiLevel(ksi, dE=0.001):
    Nkd = max(ksi.rMats['H'].shape[0:3])
    H = ksi.k_grid("H", repeat=15 // Nkd +1)
    E = np.linalg.eigvalsh( H.reshape((-1, *H.shape[3::])) )
    bandMin = np.min(E, axis=0)
    bandMax = np.max(E, axis=0)
    for lower, upper in zip(bandMin[1::], bandMax[:-1:]):
        if lower > upper + dE:
            return (lower + upper) / 2

def calcEpsilon(w, E, p, Ef, V, smearing=0.05/27.2, T=1e-3):
    eps = np.zeros((w.size, 3, 3), dtype=complex)
    if T > 0:
        focc = 1/ ( 1 + np.exp( (E-Ef)/T) )
    else:
        focc = 1 * (E < Ef)
    for a in range(E.shape[1]):
        for b in range(E.shape[1]):
            mask = (focc[:, a] < 1-1e-4) & (focc[:, b] > 1e-4) & (np.abs(focc[:, a] - focc[:, b]) > 1e-4)
            eTrans = (E[:, a] - E[:, b])[mask]
            f = focc[:, b][mask]
            pab = p[mask, a, b]
            for i, cw in enumerate(w):
                eps[i] += np.einsum("kx,ky,k->xy", pab, pab.conj(), 
                                    f/(eTrans**2 - cw**2 - 1j *smearing * cw) / eTrans)
    # impose correct normalization -- taken from epsilon.x in QE
    eps *= 128 / V / E.shape[0]
    np.einsum("kaa", eps)[:] += 1
    return eps



def evalShiftedGrid(kFrac, ksi, omega, Ef, smearing, T):
    V = np.linalg.det(ksi.lattice)
    Hk = ksi.k_grid("H", kFrac=kFrac)
    Hk = Hk.reshape((-1, *Hk.shape[3::]))
    dHk = ksi.dk_grid("H", kFrac=kFrac)
    pShape = (-1, *dHk.shape[3::])
    dHk = dHk.reshape(pShape)
    Ek, Uk = np.linalg.eigh(Hk)

    epsDict = { }
    path = None
    for vs in ['p', 'v']:
        if not vs in ksi.rMats.keys():
            continue
        v = ksi.k_grid(vs, kFrac=kFrac).reshape(pShape)
        if path is None:
            path = np.einsum_path("sba,sbcx,scd->sadx", Uk, v, Uk, optimize='optimal')[0]
        vH = np.einsum("sba,sbcx,scd->sadx", Uk.conj(), v, Uk, optimize=path)
        epsDict[vs] =  calcEpsilon(omega, Ek, vH, Ef, V, smearing, T)
    pathComm1 = None
    pathComm2 = None
    for rName in ksi.rMats.keys():
        if not rName.startswith("R"):
            continue
        pName = rName.replace("R", "v")
        D = ksi.k_grid(rName, kFrac=kFrac).reshape(pShape)
        if pathComm1 is None:
            pathComm1 = np.einsum_path("smk,sknd->smnd", Hk, D, optimize='optimal')[0]
            pathComm2 = np.einsum_path("smkd,skn->smnd", D, Hk, optimize='optimal')[0]
        vCom = 1j *(  np.einsum("smk,sknd->smnd", Hk, D, optimize=pathComm1)
                     - np.einsum("smkd,skn->smnd", D, Hk, optimize=pathComm2)) + dHk
        if path is None:
            path = np.einsum_path("sba,sbcx,scd->sadx", Uk.conj(), vCom, Uk, optimize='optimal')[0]
        vComH = np.einsum("sba,sbcx,scd->sadx", Uk.conj(), vCom, Uk, optimize=path)
        epsDict[pName] = calcEpsilon(omega, Ek, vComH, Ef, V, smearing, T)
    return epsDict


def main(seedname, dimension, enforce_hermiticity, real_wf, outFname=None, **kwargs):
    if outFname is None:
        outFname = seedname + "_eps.npz"
    data = {k : v  for k, v in np.load(seedname + "_tb.npz").items()}
    ksi = KspaceInterpolator(**data)    
    if enforce_hermiticity:
        ksi.enforce_hermiticity()
    if real_wf:
        ksi.restrict_to_real_or_imag()
    Nk = ksi.rMats['H'].shape[0:3]
    if dimension is None:
        dimension = 3
        while dimension>=0 and Nk[dimension-1] == 1:
            dimension -= 1

    omegaMin = atu.from_eV(kwargs['min'])
    omegaMax = atu.from_eV(kwargs['max'])
    dOmega = atu.from_eV(kwargs['dOmega'])
    Nk = kwargs['Nk']
    smearing = atu.from_eV(kwargs['smearing'])
    T = atu.from_K(kwargs['temperature'])

    omega = np.arange(omegaMin, omegaMax + dOmega/2, dOmega)
    Nkd = max(ksi.rMats['H'].shape[0:3])
    assert Nk >= Nkd
    while Nk % Nkd != 0:
        Nkd += 1
    expand_grid =  ( *((Nkd,) *dimension), *((1,) *(3-dimension)) )
    ksi.expand_grid(expand_grid)
    repeat = Nk // Nkd
    repeat_grid =  ( *((repeat,) *dimension), *((1,) *(3-dimension)) )

    Ef = extractFermiLevel(ksi)
    kFracOffsets = (np.mgrid[0:repeat_grid[0], 0:repeat_grid[1], 0:repeat_grid[2]] / Nk).swapaxes(0, 3).reshape((-1, 3))

    with threadpool_limits(limits=1, user_api='openmp'):
        with Pool(psutil.cpu_count(logical=False)) as p:
            res = p.map(functools.partial(evalShiftedGrid,
                                          ksi=ksi, omega=omega, Ef=Ef, smearing=smearing, T=T),
                                          [kFrac for kFrac in kFracOffsets])
    epsDict = res[0]
    for cepsDict in res[1::]:
        for k, v in cepsDict.items():
            epsDict[k] += v
    for k, eps in epsDict.items():
        eps /= kFracOffsets.shape[0]
    epsDict['omega'] = omega
    np.savez_compressed(outFname, **epsDict)



def createParser():
    parser = argparse.ArgumentParser(
                        description="""Computes the dielectric constant based on the velocity operators""",
                        epilog="from MT")
    parser.add_argument('seedname', help='Wannier90 seedname')
    parser.add_argument('-d', '--dimension', help='dimension', type=int)
    parser.add_argument('-o', '--output', help='output filename', type=str, dest='outFname')
    parser.add_argument('-H', '--enforce_hermiticity', help='make all all matrix elements in BZ Hermitian', action='store_true')
    parser.add_argument('-N', '--Nk', help="number of grid points for integration in each direction", type=int, default=40)
    parser.add_argument('-R', '--real_wf', help='neglect imaginary part of Wannier functions', action='store_true')
    parser.add_argument('-T', '--temperature', help="temperature in Kelvin", type=float, default=0)
    parser.add_argument('-s', '--smearing', help="interband smearing in eV", type=float, default=0.05)
    parser.add_argument('--min', help='minimum energy in eV', type=float, default=0.1)
    parser.add_argument('--max', help='minimum energy in eV', type=float, default=15)
    parser.add_argument('--dOmega', help='sampling rate in eV', type=float, default=0.01)
    return parser


if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args()
    main(**vars(args))

