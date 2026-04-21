#!/usr/bin/env python3

""" Visualizes the results of calcSigma """

import argparse
from matplotlib import pyplot as plt
import numpy as np

import atu

def main(seedname, direction, schemes, visType, normalize):
    assert len(direction) == 2
    d1, d2 = [ int(ord(d)-ord("x")) for d in direction]
    data = np.load(seedname + "_sigma.npz")
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    omega = data['omega']
    keys = {}
    for k in data.keys():
        if k == 'omega':
            continue
        kLast =  k.split("_")[-1] 
        if not schemes is None and not kLast in schemes:
            continue
        keys[k] = kLast
    visDict = { "Re" : dict(f=lambda eps, omega : np.real(eps), label=r"Re $\sigma_{{{}}}$ [arb.u.]"),
                "Im" : dict(f=lambda eps, omega : np.imag(eps), label=r"Im $\sigma_{{{}}}$ [arb.u.]"),
                "eps2" : dict(f=lambda eps, omega : 4*np.pi * np.real(eps)/ omega, label=r"Re $\epsilon_{{{}}}$ [arb.u.]"),
              }
    f = visDict[visType]['f']
    ylabel = visDict[visType]['label']
    norm = 1
    if normalize:
        norm =  max ( [ np.max(np.abs(f(data[k][:, d1, d2], omega))) for k in keys] )
    for k, label in keys.items():
        eps = data[k]
        ax.plot(atu.to_eV(omega), f(eps[:, d1, d2], omega) / norm, label=label)
        ax.set_xlabel("$\\omega$ [eV]")
        ax.set_ylabel(ylabel.format(direction))
    ax.legend()
    fig.tight_layout()
    plt.show()


def createParser():
    parser = argparse.ArgumentParser(
                        description="""Visualizes the optical conductivity for different calculation schemes (based on calcEpsilon)""",
                        epilog="from MT")
    parser.add_argument('seedname', help='Wannier90 seedname')
    parser.add_argument('-d', '--direction', help='e.g. xx', default="xx")
    parser.add_argument('-s', '--schemes', type=lambda s: s.split(","),
                        help=f'comma separated list of interpolation schemes to evaluate (default all)',
                        default=None)
    parser.add_argument('-t', '--type', dest='visType', default="Re", choices=["Re", "Im", "eps1", "eps2"])
    parser.add_argument('-n', '--normalize', help="normalize to 1", action='store_true')
    return parser


if __name__ == "__main__":
    parser = createParser()
    args = parser.parse_args()
    main(**vars(args))
