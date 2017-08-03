#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import statsmodels.api as sm
import statsmodels.sandbox.stats.multicomp as smm

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import seaborn as sns
current_palette = sns.color_palette("hls", 8)
sns.set_palette(current_palette)

try:
    from pyearth import Earth as earth
    USE_MARS = True
except ImportError:
    print('Cannot import pyearth, using linear_splines')
    USE_MARS = False


# Color map
CMAP = {'JJA': 'crimson', 'SON': 'darkorange',
        'DJF': 'steelblue', 'MAM': 'seagreen'}


class SeaiceSplines(object):
    """docstring for SeaiceSplines"""

    def __init__(self, data):
        super(SeaiceSplines, self).__init__()
        self.data = data
        self.nyrs = len(data)
        self.yrs = list(range(1850, self.nyrs + 1850))
        #self.preds = None
        #self.pvals = None
        #self.reject = None
        #self.pvals_corr = None

    def fit(self, knots, mars=False):
        if mars and USE_MARS:
            return self._fit_mars()
        return self._linear_splines(knots)

    def plot(self, season=None, ax=None):
        return self._plot_splines(self.preds, season=season, ax=ax)

    def knot_sig(self, method='fdr_tsbh'):
        self.reject, self.pvals_corr = smm.multipletests(
            self.pvals, method=method)[:2]
        return self.reject

    def _linear_splines(self, knots, degree=1):
        minyr = 1850

        # Use indices of years for knots
        xi_k = [k - minyr for k in knots]

        # Column vector for beta1 are (t1, t2,...,tn)
        X1 = np.arange(self.nyrs)
        X1 = X1[:, np.newaxis]

        # Generate design matrix
        X2 = np.zeros((self.nyrs, len(xi_k)))
        for col, k in enumerate(xi_k):
            X2[k:, col] = np.abs(X2[k:, col] - X1[:self.nyrs - k, 0])
        X = np.hstack((X1, X2))
        X = sm.add_constant(X)

        # Fit model, save fittedvalues and pvalues
        fit = sm.genmod.GLM(self.data, X).fit()
        self.preds = fit.fittedvalues
        self.pvals = fit.pvalues
        return fit

    def _fit_mars():
        y = self.data[:, np.newaxis]
        x = self.yrs[:, np.newaxis]

        model = earth(penalty=3, minspan_alpha=0.05, endspan_alpha=0.05)
        model.fit(x, y)
        self.preds = model.predict(x)
        self.knots = {int(bf.get_knot())
                      for bf in model.basis_ if bf.has_knot() and not bf.is_pruned()}

    def _plot_splines(self, preds, season='Season', col='g', ax=None):
        if ax is None:
            ax = plt.gca()

        spl = ax.plot(self.yrs, self.data, label='data',
                      color='k', linestyle=':')
        ax.plot(self.yrs, preds, label='trend', color=col, linewidth=2.5)
        #ax.legend(bbox_to_anchor=(.95, 0.8), loc='upper left', fontsize=14)
        ax.legend(loc='upper left', fontsize=14)
        ax.set_xlim(1845, 2016)
        ax.set_xlabel(r'Year', fontsize=17)
        ax.set_ylabel(r'Sea Ice Concentration [\%]', fontsize=17)
        title = 'Linear Splines Trend for {}'.format(season)
        ax.set_title(r'\textbf{' + title + '}', fontsize=19)
        ax.set_xticks(np.linspace(1850, 2013, 9, dtype=int))
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
        return spl
