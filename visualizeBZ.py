#!/usr/bin/env python3

""" Visuzliation (through sliders) of interpolated Hamiltonian, Berry connection
and optionally velocity matrix elements.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


from KspaceInterpolator import KspaceInterpolator
import KspaceGUI


def createParser():
    parser = argparse.ArgumentParser(
                        description="""Visualizes Berry connection and velocity and operator in BZ""",
                        epilog="from MT")
    parser.add_argument('seedname', help='Wannier90 seedname')
    parser.add_argument('-e', '--expand_grid', help='expands FFT grid real space grid to given number', type=int)
    parser.add_argument('-r', '--repeat_grid', help='repeats FFT grid in each dimension given number of times during display',
                                               type=int, default=1)
    parser.add_argument('-H', '--enforce_hermiticity', help='make all all matrix elements in BZ Hermitian', action='store_true')
    parser.add_argument('-R', '--real_wf', help='neglect imaginary part of Wannier functions', action='store_true')
    return parser



def main(seedname, enforce_hermiticity, real_wf, expand_grid, repeat_grid):
    data = {k : v  for k, v in np.load(seedname + "_tb.npz").items()}
    ksi = KspaceInterpolator(**data)    
    if enforce_hermiticity:
        ksi.enforce_hermiticity()
    if real_wf:
        ksi.restrict_to_real_or_imag()
    if expand_grid:
        ksi.expand_grid(expand_grid)
    Mks = ksi.k_grid(repeat=repeat_grid)
    Hk = Mks.pop("H")
    dHk = ksi.dk_grid("H", repeat=repeat_grid)

    for k in data.keys():
        if not k.startswith("R_"):
            continue
        vKcom = 1j *(  np.einsum("abcmk,abcknd->abcmnd", Hk, Mks[k])
                     - np.einsum("abcmkd,abckn->abcmnd", Mks[k], Hk)) + dHk
        Mks[k.replace("R", "v")] = vKcom

    gui = KspaceGUI.KspaceGUI(Hk, units="a.u", sublabels={"R_" : "R", "v_" : "v"}, **Mks)
    plt.show()


if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args()
    main(**vars(args))

