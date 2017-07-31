#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
from argparse import ArgumentParser
from datetime impor datetime

import numpy as np
import xarray as xr
import pandas as pd

from scipy import signal
import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.sandbox.stats.multicomp as smm
import statsmodels.tsa.stattools as tsa
import statsmodels.tsa.arima_model as arima
from statsmodels.stats.diagnostic import acorr_ljungbox as ljungbox

import matplotlib.pyplot as plt
import seaborn as sns
current_palette = sns.color_palette("hls", 8)
sns.set_palette(current_palette)

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

# Seaice data set
DATA_DIR = os.path.join(os.getcwd(), 'data')
DATA = '{}/G10010_SIBT1850_v1.1.nc'.format(DATA_DIR)
SDF = {'DJF': '{}/djf_df.csv'.format(DATA_DIR),
       'MAM': '{}/mam_df.csv'.format(DATA_DIR),
       'JJA': '{}/jja_df.csv'.format(DATA_DIR),
       'SON': '{}/son_df.csv'.format(DATA_DIR)}

# Color map
CMAP = {'JJA': 'crimson', 'SON': 'darkorange',
        'DJF': 'steelblue', 'MAM': 'seagreen'}


class SeaiceSplines(object):
    """docstring for SeaiceSplines"""

    def __init__(self, spatial=False):
        super(SeaiceSplines, self).__init__()

        self.nc = xr.open_dataset(DATA)
        self.spatial = spatial

        if self.spatial:

        self.seasons = ['DJF', 'MAM', 'JJA', 'SON']
        self.dfs = {season: pd.read_csv(df) for season, df in SDF.items()}
        self.knots = list()

    def _linear_splines(data, knots, degree=1):
        minyr = data.year.min()
        nyrs = data.year.shape[0]

        # Use indices of years for knots
        xi_k = [k - minyr for k in knots]

        # Column vector for beta1 are (t1, t2,...,tn)
        X1 = np.arange(nyrs)
        X1 = X1[:, np.newaxis]

        # Generate design matrix
        X2 = np.zeros((nyrs, len(xi_k)))
        for col, k in enumerate(xi_k):
            X2[k:, col] = np.abs(X2[k:, col] - X1[:nyrs - k, 0])
        X = np.hstack((X1, X2))
        return smf.glm('seaice_conc ~ X', data=data).fit()

    def fit(season, knots):
        df = SDF[season.upper()]
        return _linear_splines(df, knots)


def plot_corr(acf_x, nlags, ax=None, col='r', title='Autocorrelation'):
    def bonferroni_confint(x, alpha=0.05):
        nobs = len(x)
        confint = sp.stats.norm.ppf(1 - alpha / nobs) / np.sqrt(nobs)
        return confint, -confint

    confint = bonferroni_confint(acf_x)

    if ax is None:
        ax = plt.gca()

    fig = ax.stem(acf_x, color=col)
    plt.setp(fig[1], color=col)
    plt.setp(fig[0], 'markersize', 0)
    ax.set_title(r"\textbf{" + title + "}", fontsize=19)
    ax.set_ylim([min(acf_x) - 0.3, 1])
    ax.set_xlabel(r"Lag", fontsize=17)

    if "pacf" in title.lower() or "partial" in title.lower():
        ax.set_ylabel(r"Partial ACF", fontsize=17)
    else:
        ax.set_ylabel(r"ACF", fontsize=17)

    if confint is not None:
        ax.axhline(y=confint[0], xmin=0, xmax=nlags,
                   c='black', linewidth=0.5, linestyle='--', zorder=0)
        ax.axhline(y=confint[1], xmin=0, xmax=nlags,
                   c='black', linewidth=0.5, linestyle='--', zorder=0)
    return fig


def plot_acf(data, nlags, alpha=0.05, ax=None, col='r', title='ACF'):
    acf_x, _ = tsa.acf(data, nlags=nlags, alpha=alpha)
    return plot_corr(acf_x, nlags=nlags, ax=ax, col=col, title=title)


def plot_pacf(data, nlags, alpha=0.05, ax=None, col='r', title='PACF'):
    method = 'ywm'
    pacf_x, _ = tsa.pacf(data, nlags=nlags, alpha=alpha, method=method)
    return plot_corr(pacf_x, nlags=nlags, ax=ax, col=col, title=title)


def test_stationarity(timeseries, label='Original', ax=None):
    if ax is None:
        ax = plt.gca()

    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=15)
    rolstd = pd.rolling_std(timeseries, window=15)

    # Plot rolling statistics:
    orig = ax.plot(timeseries, color='k', alpha=0.3, label=label)
    mean = ax.plot(rolmean, color='darkred', label='Rolling Mean')
    std = ax.plot(rolstd, color='darkorange', label='Rolling Std')
    ax.legend(loc='best')
    ax.set_title('Rolling Mean & Standard Deviation')
    # ax.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = tsa.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic',
                         'p-value',
                         '#Lags Used',
                         'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def ts_diagnostics(fit, col='r', title='Season', save=None):
    e_ar = fit.resid
    x = np.arange(len(e_ar))

    import matplotlib.gridspec as gridspec
    f = plt.figure(figsize=(14, 14))
    sns.set_style("ticks")
    gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    ax4 = plt.subplot(gs[2, 0])
    ax5 = plt.subplot(gs[2, 1])

    ax1.plot(list(range(1850, 2014)), e_ar, color=col)
    r"\textbf{" + title + "}"
    ax1.set_title(r'\textbf{' + title + ' Residuals}', fontsize=19)
    ax1.set_xticks(np.linspace(1850, 2013, 9, dtype=int))
    ax1.set_xlabel(r'Year', fontsize=17)
    ax1.xaxis.set_tick_params(labelsize=13)
    ax1.yaxis.set_tick_params(labelsize=13)

    plot_acf(e_ar, nlags=len(e_ar) - 1, col=col,
             title=title + " ACF", ax=ax2)
    ax2.xaxis.set_tick_params(labelsize=13)
    ax2.yaxis.set_tick_params(labelsize=13)

    plot_pacf(e_ar, nlags=len(e_ar) - 1, col=col,
              title=title + " PACF", ax=ax3)
    ax3.xaxis.set_tick_params(labelsize=13)
    ax3.yaxis.set_tick_params(labelsize=13)

    lb = ljungbox(e_ar, 10)[1]
    ax4.scatter(list(range(1, 11)), lb, color=col)
    ax4.set_ylim(min(lb) - 0.05, max(lb) + 0.05)
    yticks = np.linspace(min(lb), max(lb), 6)
    ax4.set_yticks(np.round(yticks, 3))
    ax4.set_title(
        r'\textbf{P-values for Ljung-Box statistic}', fontsize=19)
    ax4.set_xlabel(r'Lag', fontsize=17)
    ax4.xaxis.set_tick_params(labelsize=13)
    ax4.yaxis.set_tick_params(labelsize=13)

    stats.probplot(e_ar, dist="norm", plot=ax5)
    ax5.set_title(r'\textbf{QQ Plot of Residuals}', fontsize=19)
    ax5.set_xlabel(r'Theoretical quantiles', fontsize=17)
    ax5.set_ylabel(r'Ordered Values', fontsize=17)
    ax5.xaxis.set_tick_params(labelsize=13)
    ax5.yaxis.set_tick_params(labelsize=13)

    f.subplots_adjust(wspace=0.25, hspace=0.5)
    f.tight_layout()
    sns.despine()

    if save:
        fname = '../writeup/figs/{}.eps'.format(save)
        f.savefig(fname, format='eps', dpi=1200, bbox_inches='tight')
    return e_ar


def main():
    s = SeaiceSplines()


if __name__ == '__main__':
    main()
