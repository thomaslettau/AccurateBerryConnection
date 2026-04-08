#!/usr/bin/env python3

import sys
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib import colors

class KspaceGUI:
    def __init__(self, Hk, units="??", sublabels={}, **kwargs):
        self.Hk = Hk
        self.mats = kwargs
        if isinstance(units, dict):
            self.units = units
        else:
            defaultUnit = units
            self.units = {"H" : defaultUnit} | { k : defaultUnit for k in kwargs.keys() }

        self.Nk1, self.Nk2, self.Nk3, self.Nw, _, = Hk.shape
        self.mat_changed(next(iter(kwargs)), initial=True)

        self.fig, self.ax = plt.subplots(2, 4, figsize=(12, 10))
        self.fig.subplots_adjust(bottom=0.25, left=0.08, right=0.98, top=0.98, wspace=0.35, hspace=0)

        self.clickId = self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_mousePressed(event))
       
        axN = self.fig.add_axes([0.25, 0.12, 0.6, 0.03])
        self.slider_n = Slider(ax=axN, label="N ", valmin=0, valmax=self.Nw-1,
                                              valstep=1, valinit=0)
        self.slider_n.on_changed(lambda event : self.updateImgsSlot(event, True))

        axM = self.fig.add_axes([0.25, 0.16, 0.6, 0.03])
        self.slider_m = Slider(ax=axM, label="M ", valmin=0, valmax=self.Nw-1,
                                              valstep=1, valinit=0)
        self.slider_m.on_changed(lambda event : self.updateImgsSlot(event, True))

        axSliceDir = self.fig.add_axes([0.1, 0.04, 0.05, 0.15])
        self.sliceDir = 0
        self.sliceDirs = [(2, 3), (1, 3), (1, 2)]
        self.sliceDirNames = [ f"{a},{b}" for (a, b) in self.sliceDirs]
        self.rb_sliceDir = RadioButtons(axSliceDir, self.sliceDirNames)
        self.rb_sliceDir.on_clicked(lambda event : self.updateSliceAxisSlot(event))

        self.axNk = self.fig.add_axes([0.25, 0.20, 0.6, 0.03])
        self.createNkSlider()

        self.button_keys = {}
        keyCount = len(kwargs.keys())
        if keyCount > 1:
            remKeys = set(kwargs.keys())
            keyList = []
            for start, label in sublabels.items():
                subkeys = [s for s in remKeys if s.startswith(start)]
                if len(subkeys) >= 1:
                    remKeys -= set(subkeys)
                    keyList.append([start, label, [s[len(start):] for s in subkeys] ])
            if len(remKeys) > 0:
                keyList.append(["","General", remKeys])

            for lline, (start, label, subkeys) in enumerate(keyList):
                h = 0.05 - 0.02 * lline
                count = len(subkeys)
                self.fig.text(0.24, h, label, horizontalalignment='right')
                for i, key in enumerate(sorted(subkeys)):
                    if count > 1:
                        s = count / (count*count-1)
                        axKey = self.fig.add_axes([0.25 + 0.6 * i * s, h, 0.6 / (count+1), 0.015])
                    else:
                        axKey = self.fig.add_axes([0.25, h, 0.6, 0.015])
                    self.button_keys[start+key] = Button(axKey, label=key)
                    self.button_keys[start+key].on_clicked(lambda event, key=start+key: self.mat_changed(key))



        axReset = self.fig.add_axes([0.25, 0.08, 0.6, 0.03])
        self.button_reset = Button(axReset, label="Reset")
        self.button_reset.on_clicked(lambda event : self.reset(event))

        self.cbars = []

        self.initImgs()
        self.updateSliceAxis()

        self.isp = False # in signal processing


    def createNkSlider(self):
        newNkSlice = [self.Nk1, self.Nk2, self.Nk3][self.sliceDir]
        if hasattr(self, "slider_Nk") and self.NkSlice == newNkSlice:
            return
        self.axNk.clear()
        self.NkSlice = newNkSlice
        self.slider_Nk = Slider(self.axNk, label="k-Slice: ", valmin=self.bMin, valmax=1+self.bMin, valinit=self.bMin+0.5,
                                              valstep=np.linspace(self.bMin, self.bMin+1, self.NkSlice+1))
        self.slider_Nk.on_changed(lambda event : self.updateImgsSlot(event))

    def reset(self, event):
        if self.isp:
            return
        self.isp = True
        self.slider_n.reset()
        self.slider_m.reset()
        self.slider_Nk.reset()
        self.rb_sliceDir.set_active(0)
        self.updateSliceAxis()
        self.updateImgs()
        self.isp = False

    def mat_changed(self, key, initial=False):
        M = self.mats[key]
        self.operators = np.stack((self.Hk, M[...,0], M[...,1], M[...,2]))
        if "_" in key:
            before, after = key.split("_")
            magF = "$|" + before + "_\\mathrm{{" + after + ",{}}}|$"
            phaseF = "$\\arg(" + before + "_\\mathrm{{" + after + ",{}}})$"
        else:
            magF = "$|" + key + "_{}|$"
            phaseF = "$\\arg(" + key + "_{})$"
        magF = magF + f" [{self.units[key]}]"
        self.opMagLabels = [ r"$|H|$" + f"[{self.units['H']}]"] + [magF.format(s) for s in "xyz"]
        self.opPhaseLabels = [ r"$\arg(H)$"] + [ phaseF.format(s) for s in "xyz"]
        self.bMin = -0.5
        if not ((self.bMin * self.Nk1).is_integer() and (self.bMin * self.Nk2).is_integer() and (self.bMin * self.Nk3).is_integer()):
            self.bMin = 0.0
        self.operators = np.roll(self.operators, 
                                 (round(self.bMin*self.Nk1), round(self.bMin * self.Nk2), round(self.bMin * self.Nk3)),
                                 (1, 2, 3))
        if not initial:
            self.initImgs()
            self.updateSliceAxis()
            self.updateImgs()

    def updateSliceAxisSlot(self, event):
        if self.isp:
            return
        self.isp = True
        self.updateSliceAxis()
        self.updateImgs()
        self.isp = False

    def updateImgsSlot(self, event, cbRangeUpdate=False):
        if self.isp:
            return
        self.isp = True
        self.updateImgs(cbRangeUpdate)
        self.isp = False

    def getSlicedOps(self):
        Nks = round((self.slider_Nk.val+1) * self.NkSlice) % self.NkSlice
        n = int(self.slider_n.val)
        m = int(self.slider_m.val)
        if self.sliceDir == 0:
            return self.operators[:, Nks, :, :, m, n]
        elif self.sliceDir == 1:
            return self.operators[:, :, Nks, :, m, n]
        elif self.sliceDir == 2:
            return self.operators[:, :, :, Nks, m, n]

    def initImgs(self):
        n = int(self.slider_n.val)
        m = int(self.slider_m.val)
        self.ims = []
        for c in self.cbars:
            c.remove()
        self.cbars = []
        slicedOps = self.getSlicedOps()
        for e, (axAbs, axPhase) in enumerate(self.ax.T):
            norm = colors.Normalize(vmin=0, vmax=np.max(np.abs(self.operators[e, :, :, :, m, n] )))
            imAbs = axAbs.imshow(np.abs(slicedOps[e, :, :]), extent=[self.bMin, self.bMin+1, self.bMin, self.bMin+1])
            imAbs.set_norm(norm)
            cbar = self.fig.colorbar(imAbs, ax=axAbs, orientation='horizontal')
            cbar.set_label(self.opMagLabels[e])
            self.cbars.append(cbar)
            imPhase = axPhase.imshow(np.angle(slicedOps[e, :, :]), cmap='twilight', extent=[self.bMin, self.bMin+1, self.bMin, self.bMin+1])
            imPhase.set_norm(colors.Normalize(vmin=-np.pi, vmax=np.pi))
            cbar = self.fig.colorbar(imPhase, ax=axPhase, orientation='horizontal')
            cbar.set_label(self.opPhaseLabels[e])
            self.cbars.append(cbar)
            self.ims.append([imAbs, imPhase])


    def updateSliceAxis(self):
        self.sliceDir = self.sliceDirNames.index(self.rb_sliceDir.value_selected)
        self.createNkSlider()
        d1, d2 = self.sliceDirs[self.sliceDir]
        for ax in self.ax.flat:
            ax.set_ylabel(r"$b_{{{}}}$".format(d1))
            ax.set_xlabel(r"$b_{{{}}}$".format(d2))

    def updateImgs(self, cbRangeUpdate=False):
        n = int(self.slider_n.val)
        m = int(self.slider_m.val)
        slicedOps = self.getSlicedOps()
        for e, [imAbs, imPhase] in enumerate(self.ims):
            if cbRangeUpdate:
                imAbs.set_clim(vmin=0, vmax=np.max(np.abs(self.operators[e, :, :, :, m, n])))
            imAbs.set_data(np.abs(slicedOps[e, :, :]))
            imPhase.set_data(np.angle(slicedOps[e, :, :]))
        self.fig.canvas.draw()

    def setTitle(self, title):
        self.fig.canvas.manager.set_window_title(title)

    def on_mousePressed(self, event):
        if event.dblclick:
            for e, [imOrigAbs, imOrigPhase] in enumerate(self.ims):
                if event.inaxes in [imOrigAbs.axes, imOrigPhase.axes]:
                    opMatrix = self.getSlicedOps()[e, :, :]
                    popupFig, (axAbs, axPhase)  = plt.subplots(1, 2)
                    imAbs = axAbs.imshow(np.abs(opMatrix), extent=[self.bMin, self.bMin+1, self.bMin, self.bMin+1])
                    cbar = popupFig.colorbar(imAbs, ax=axAbs, orientation='horizontal')
                    cbar.set_label(self.opMagLabels[e])
                    angleSign = 1 if event.inaxes == imOrigAbs.axes else -1
                    imPhase = axPhase.imshow(angleSign * np.angle(opMatrix), cmap='twilight', extent=[self.bMin, self.bMin+1, self.bMin, self.bMin+1])
                    imPhase.set_norm(colors.Normalize(vmin=-np.pi, vmax=np.pi))
                    cbar = popupFig.colorbar(imPhase, ax=axPhase, orientation='horizontal')
                    cbar.set_label(self.opPhaseLabels[e])

                    d1, d2 = self.sliceDirs[self.sliceDir]
                    for ax in [axAbs, axPhase]:
                        ax.set_ylabel(r"$b_{{{}}}$".format(d1))
                        ax.set_xlabel(r"$b_{{{}}}$".format(d2))
                   
                    n = int(self.slider_n.val)
                    m = int(self.slider_m.val)
                    invPhaseComment = ", Phase inverted" if angleSign == -1 else ""
                    popupFig.suptitle(f"m={m}, n={n}, b{e}={round(self.slider_Nk.val, 4)}" + invPhaseComment)
                    popupFig.tight_layout()
                    plt.show()
