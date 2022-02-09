""" This file contains different possible degree functions """
import numpy as np


class SparseDeg:
    def __init__(self, maxdeg: dict, wn=1.0, wl=1.5):
        self.maxdeg = maxdeg
        self.wn = wn
        self.wl = wl

    def __call__(self, n: int, l: int, corr: int):
        return self.wn * np.sum(n) + self.wl * np.sum(l) <= self.maxdeg[corr]

    def max_n(self):
        max_deg = max(self.maxdeg.values())
        return int(max_deg // self.wn)

    def max_l(self):
        max_deg = max(self.maxdeg.values())
        return int(max_deg // self.wl)

    def max_corr(self):
        return int(len(self.maxdeg))


class NaiveMaxDeg:
    def __init__(self, maxdeg: dict):
        # maxdeg is a dictionary containing nmax and lmax for each correlation order
        self.maxdeg = maxdeg

    def __call__(self, n: int, l: int, corr: int):
        return self.maxdeg[corr][0] >= np.sum(n) and self.maxdeg[corr][1] >= np.sum(l)//corr

    def max_n(self):
        nl_s = np.array(list(self.maxdeg.values()))
        return int(max(nl_s[:, 0]))

    def max_l(self):
        nl_s = np.array(list(self.maxdeg.values()))
        return int(max(nl_s[:, 1]))

    def max_corr(self):
        return int(len(self.maxdeg))
