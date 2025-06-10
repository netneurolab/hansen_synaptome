"""
code for interpreting time-series feature hits
and testing specificity of synapse-hctsafeature relationships
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from palettable.colorbrewer.sequential import (PuBuGn_4,
                                               PuRd_4,
                                               PuBuGn_8)
from scipy.stats import spearmanr, zscore
from statsmodels.stats.multitest import multipletests
from ast import literal_eval
import mat73
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AgglomerativeClustering
from netneurotools import plotting


def scatter_types(x, y, ont_names, cmap_ontology, ax, ont_names_inv=None):
    for idx, (name, colour) in enumerate(zip(ont_names,
                                             cmap_ontology)):
        mask = ont_names_inv == idx
        ax.scatter(x=x[mask], y=y[mask], label=name, color=colour)


def regress_out(y, X):

    """
    Regress out the effects of multiple variables (X)
    from a target variable (y).

    Parameters:
        y (array-like): Target variable (n_samples,).
        X (array-like): Predictor variables (n_samples, n_predictors).

    Returns:
        residuals (array): The target variable with the effects of X
        regressed out.
    """

    X = np.asarray(X)  # Ensure X is a NumPy array
    if X.ndim == 1:
        X = X.reshape(-1, 1)  # Handle single predictor case

    model = LinearRegression()
    model.fit(X, y)  # Fit the model using multiple predictors
    y_pred = model.predict(X)  # Predicted values based on X
    residuals = y - y_pred  # Subtract predicted values to remove X's effects

    return residuals


def SY_StatAv(y, whatType='seg', n=5):

    """
    SY_StatAv: Simple mean-stationarity metric, StatAv.

    This function divides a time series into non-overlapping subsegments,
    calculates the mean in each of these segments, and returns the standard
    deviation of these means, normalized by the standard deviation of the
    original time series.

    This code is translated from Ben Fulcher's Matlab code.

    Parameters:
        y (array-like): Input time series.
        whatType (str): Type of StatAv to perform:
                        - 'seg': Divide the time series into n segments.
                        - 'len': Divide the time series into segments of
                                 length n.
        n (int): Number of segments ('seg') or segment length ('len').

    Returns:
        float: StatAv (normalized standard deviation of segment means).
    """

    y = np.asarray(y)  # Ensure y is a NumPy array
    N = len(y)  # Length of the time series

    if whatType == 'seg':
        # Divide the time series into n equal segments
        p = N // n  # Segment length
        if p < 1:
            raise ValueError("Number of segments (n) is too large for the\
                             time series length.")
        M = [np.mean(y[p * j:p * (j + 1)]) for j in range(n)]  # Segment means

    elif whatType == 'len':
        # Divide the time series into segments of length n
        if N > 2 * n:
            pn = N // n  # Number of complete segments
            M = [np.mean(y[j * n:(j + 1) * n]) for j in range(pn)]  # means
        else:
            print(f"This time series (N = {N}) is too short\
                  for StatAv('{whatType}', {n}).")
            return np.nan

    else:
        raise ValueError("Invalid 'whatType'.\
                         Please select either 'seg' or 'len'.")

    # Compute the StatAv statistic
    s = np.std(y, ddof=1)  # Standard deviation of the original time series
    sdav = np.std(M, ddof=1)  # Standard deviation of the segment means
    out = sdav / s  # Normalize by std of the original time series

    return out


def plot_binned_means(time_series, n_bins, set_axes=False):

    """
    Plot a time series, bin it into n equal segments, and plot the bin means
    normalized by std of entire time series, centered at bin midpoints.

    Parameters:
        time_series (array-like): The input time series.
        n_bins (int): The number of bins to divide the time series into.
    """

    time_series = np.asarray(time_series)
    N = len(time_series)
    bin_size = N // n_bins  # Size of each bin

    if bin_size < 1:
        raise ValueError("Number of bins is too large\
                         for the time series length.")

    # Compute bin edges, bin midpoints, and bin means
    bin_edges = [i * bin_size for i in range(n_bins + 1)]
    bin_midpoints = [(bin_edges[i] + bin_edges[i + 1]) / 2
                     for i in range(n_bins)]  # Centers of bins
    bin_means = [np.mean(time_series[bin_edges[i]:bin_edges[i + 1]])
                 for i in range(n_bins)]

    # Create the figure
    fig = plt.figure(figsize=(10, 8))

    # Plot 1: Time series with bin divisions
    ax_ts = fig.add_subplot(211)  # Top subplot
    ax_ts.plot(time_series, color='black', label='Time series')
    for edge in bin_edges[1:-1]:  # Skip first and last edges
        ax_ts.axvline(edge, color='red', linestyle='--', alpha=0.7)
    ax_ts.set_title("Time Series with Bin Divisions")
    ax_ts.set_ylabel("Amplitude")
    ax_ts.set_xlabel("Time")
    ax_ts.legend()

    # Plot 2: Means of each bin centered at bin midpoints
    ax_means = fig.add_subplot(212)  # Bottom subplot
    ax_means.plot(bin_midpoints,
                  bin_means/np.std(time_series, ddof=1),
                  'o-', color='blue',
                  label='Bin Means')
    ax_means.set_title("Mean Values of Each Bin (Centered)")
    ax_means.set_ylabel("Mean Value")
    ax_means.set_xlabel("Time (Bin Midpoints)")
    ax_means.legend()

    if set_axes:
        ax_ts.set_ylim(-4.5, 5)
        ax_ts.set_xlim(-50, 1450)
        ax_means.set_ylim(-0.15, 0.2)
        ax_means.set_xlim(-50, 1450)

    plt.tight_layout()
    plt.show()

    return fig


def plot_outlier_include(y, threshold_how='n', inc=0.01):

    """
    Visualize how DN_OutlierInclude computes out.mrmd in Python.

    Parameters:
        y (array-like): Input time series (ideally z-scored).
        threshold_how (str): Method to define outliers ('n' for neg).
        inc (float): Increment for threshold values (e.g., 0.01).
    """

    # Ensure the time series is z-scored
    y = (y - np.mean(y)) / np.std(y) if np.std(y) != 0 else y

    # Length of the time series
    N = len(y)

    # Rescale x-axis: middle is 0, first is -1, last is 1
    x_rescaled = (np.arange(N) / (N / 2)) - 1

    # Define thresholds for negative deviations
    if threshold_how == 'n':
        thr = np.arange(0, max(-y) + inc, inc)
    else:
        raise ValueError("Unsupported threshold_how. Only 'n'\
                         (negative deviations) is implemented.")

    # Prepare to store median timings
    msDt = []

    # Create a figure for the plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8),
                            gridspec_kw={'height_ratios': [2, 1]})

    # Iterate through thresholds
    for i, th in enumerate(thr):
        # Find points below the negative threshold
        r = np.where(y <= -th)[0]

        if len(r) == 0:
            msDt.append(None)
        else:
            # Compute mean timing relative to the middle (N/2)
            msDt.append((np.mean(r) / (N / 2)) - 1)

    # Plot the time series and included points for some th
    th = thr[np.int(2/3 * len(thr))]
    r = np.where(y <= -th)[0]
    axs[0].plot(x_rescaled, y, '-k', linewidth=1.5, label='Time Series')
    axs[0].plot(x_rescaled[r], y[r], 'or',
                markersize=6, label='Included Points')
    axs[0].set_title(f'Time Series and Included Points (Threshold: {-th:.2f})')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[0].legend(loc='best')
    axs[0].grid(True)

    axs[1].plot(thr[:i+1], msDt[:i+1], '-b', linewidth=1.5,
                label='Mean Timing')
    axs[1].scatter(th, msDt[np.int(2/3 * len(thr))], color='red',
                   s=80, label='Example Mean Timing')
    axs[1].set_title('Mean Timing (Relative to Middle) vs. Threshold')
    axs[1].set_xlabel('Threshold')
    axs[1].set_ylabel('Mean Timing')
    axs[1].grid(True)
    axs[1].legend(loc='best')

    # median mean timing (out.mrmd)
    mrmd = np.median(np.array(msDt[:-1]))

    # Add a horizontal line for the final mrmd value
    axs[1].axhline(mrmd, color='green', linestyle='--', linewidth=1.5,
                   label=f'Median Mean Timing: {mrmd:.3f}')
    axs[1].legend(loc='best')

    plt.tight_layout()
    plt.show()

    return fig, axs


path = "/home/jhansen/gitrepos/hansen_synaptome/"

"""
set-up
"""

fcregions = pd.read_excel(path+'data/function/Gozzi/' +
                          'rois_id_acr_names_N_182_ORDER_and_Exclusions.xlsx',
                          sheet_name="Exclusions")
fcregions = fcregions[fcregions['REMOVED?'] != 1]
fcregions.reset_index(drop=True, inplace=True)
fcregions = pd.concat([fcregions, fcregions], ignore_index=True)
ts = mat73.loadmat(path+'data/function/Gozzi/BOLD_timeseries_Awake.mat')[
                       'BOLD_timeseries_Awake']

# region_mapping
region_mapping = pd.read_csv(path+'data/region_mapping_fc.csv', index_col=0,
                             converters={'synaptome_acr': literal_eval,
                                         'synaptome_idx': literal_eval})
# lambda function for 88/162 regions
lfunc = region_mapping['synaptome_acr'].apply(lambda x: len(x) != 0)

# idx to sort 88 regions by ontology
ont_idx = region_mapping[lfunc].sort_values(by='ontology').index
(ont_names, ont_names_idx,
 ont_names_inv) = np.unique(fcregions['MACRO'].values[ont_idx],
                            return_index=True,
                            return_inverse=True)

# colourmap for scatterplots where points are regions
cmap_ontology = np.array([[0.39607843, 0.76862745, 0.82352941, 1.0],
                          [0.36470588, 0.63137255, 0.69019608, 1.0],
                          [0.32549020, 0.47843137, 0.65098039, 1.0],
                          [0.76862745, 0.65882353, 0.81568627, 1.0],
                          [0.52941176, 0.54117647, 0.68235294, 1.0],
                          [0.76862745, 0.88627451, 0.73725490, 1.0]
                          ])

# load synapse type density
(type1, type1l, type1s, type2,
 type3, type3c1, type3c2) = np.load(path
                                    + 'data/synaptome/mouse_liu2018/'
                                    + 'type_densities_88.npz'
                                    ).values()

# make diverging colourmap
teals = PuBuGn_4.mpl_colors
teals.reverse()
reds = PuRd_4.mpl_colors
teals.extend(reds)
teals[0] = PuBuGn_8.mpl_colors[-1]
del teals[4]
cmap_div = LinearSegmentedColormap.from_list('cmap', teals, N=256)

# load hctsa features
hctsapath = path + 'data/function/Gozzi/HCTSA/'
hctsa = dict([])
for state in ['Awake', 'Halo', 'MedIso']:
    matrix = mat73.loadmat(hctsapath + 'HCTSA_normalized_zscored_noexcl_{}.mat'
                           .format(state))['sharedTS']
    hctsa[state] = np.mean(matrix[ont_idx, :, :], axis=2)
features = pd.read_excel(hctsapath + 'HCTSA_normalized_features_noexcl.xlsx',
                         index_col=0).reset_index()

# load time-series
ts = mat73.loadmat(path+'data/function/Gozzi/BOLD_timeseries_Awake.mat')[
                       'BOLD_timeseries_Awake']

# load framewise displacement
fd = np.loadtxt(path+'data/function/Gozzi/FD_scrubbed_Awake.csv',
                delimiter=',')

"""
cluster hctsa features for interpretation
"""

# load hctsa results
mat = np.load(path+'results/HCTSA/'
              + 'hctsa_norm_zscored_noexcl-corrs_Awake.npz')
rhos = mat['rhos']
pvals = mat['pvals_corrected']

types = ['1l', '1s', '2', '3c1', '3c2']

# for each synapse type
for stype in range(rhos.shape[0]):

    # get features that are significant
    if types[stype] == '2' or \
       types[stype] == '3c1':
        sig = np.where((pvals[stype, :, 0] < 0.05) &
                       (abs(rhos[stype, :, 0]) >= 0.5))[0]
    else:
        sig = np.where(pvals[stype, :, 0] < 0.05)[0]

    if len(sig) < 10:
        continue

    # feature matrix of hits
    featMat = hctsa['Awake'][:, sig]

    # feature similarity matrix
    dataMat = zscore(featMat)
    dataMat = np.abs(spearmanr(dataMat)[0])

    # define number of clusters etc
    nnode, nfeat = dataMat.shape
    num_clusters = np.arange(2, 11)
    allLabels = np.zeros((len(num_clusters), nfeat))

    # clustering analysis
    for clustering in range(len(num_clusters)):
        nclust = num_clusters[clustering]
        clusteringResult = AgglomerativeClustering(n_clusters=nclust,
                                                   linkage='average').fit(
                                                   dataMat.T)
        allLabels[clustering, :] = clusteringResult.labels_

    # plot and save all solutions
    for solutionid in range(allLabels.shape[0]):
        nclusters = len(np.unique(allLabels[solutionid, :]))
        print(nclusters)

        # communities = np.asarray(allLabels[solutionid, :])
        communities = allLabels[solutionid, :].flatten().astype(int)
        if 0 in communities:
            communities = communities + 1

        myplot = plotting.plot_mod_heatmap(dataMat, communities,
                                           cmap=PuRd_4.mpl_colormap,
                                           vmin=np.min(dataMat), vmax=1,
                                           rasterized=True, figsize=(10, 10),
                                           xticklabels=features[
                                               'Name'].iloc[sig].values)
        plt.tight_layout()
        plt.savefig(path + 'figures/eps/'
                    + 'heatmap_featuresimilarity_type%s_nclusters%s.eps'
                    % (['1l', '1s', '2', '3c1', '3c2'][stype], nclusters),
                    bbox_inches='tight')

        # plot the absolute correlation coefficients too
        inds = plotting.sort_communities(dataMat, communities)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(np.arange(len(sig)), abs(rhos[stype, sig[inds], 0]))
        ax.set_xticks(np.arange(len(sig)))
        ax.set_xticklabels(features['Name'].iloc[sig[inds]].values,
                           rotation=90)
        ax.set_title('type{}: nclusters={}'.format(types[stype], nclusters))
        fig.tight_layout()
        fig.savefig(path + 'figures/eps/scatter_absrhos_type{}_nclusters{}.eps'
                    .format(types[stype], nclusters))

"""
dig into StatAv
"""

# pick two time-series and plot distribution of means
mouse_idx = 8

fig = plot_binned_means(zscore(ts[mouse_idx][0][np.where(
    region_mapping['fc81_acr'] == 'EP')[0][0], :]), 10, set_axes=True)
fig.savefig(path+'figures/eps/binnedmeans_mouse8_EP.eps')
fig = plot_binned_means(zscore(ts[mouse_idx][0][np.where(
    region_mapping['fc81_acr'] == 'SSp-bfd')[0][0], :]), 10, set_axes=True)
fig.savefig(path+'figures/eps/binnedmeans_mouse8_SSp-bfd.eps')

"""
dig into OutlierInclude
"""

mouse_idx = 1
fig, ax = plot_outlier_include(zscore(ts[mouse_idx][0][np.where(
    region_mapping['fc81_acr'] == 'MBmot')[0][0], :]))
fig.savefig(path+'figures/eps/outlierinclude_mouse1_MBmot.eps')
fig, ax = plot_outlier_include(zscore(ts[mouse_idx][0][np.where(
    region_mapping['fc81_acr'] == 'PAA')[0][0], :]))
fig.savefig(path+'figures/eps/outlierinclude_mouse1_PAA.eps')

# get feature value without normalization to interpret sign
matrix = mat73.loadmat(hctsapath +
                       'HCTSA_zscored_noexcl_Awake.mat')['sharedTS']
hctsa['Awake-nonorm'] = np.mean(matrix[ont_idx, :, :], axis=2)
features = pd.read_excel(hctsapath + 'HCTSA_features.xlsx',
                         index_col=0).reset_index()
i = features.query("Name == 'DN_OutlierInclude_n_001_mrmd'").index[0]
plt.figure()
plt.scatter(type2, hctsa['Awake-nonorm'][:, i])

"""
SNR
"""

snr = region_mapping[lfunc].sort_values(by='ontology')['SNR'].values

# find hctsa features that are significantly corr with SNR
snr[np.isnan(snr)] = np.nanmean(snr)
snrrhos = np.zeros((hctsa['Awake'].shape[1], 2))
for i in range(hctsa['Awake'].shape[1]):
    snrrhos[i, :] = spearmanr(snr, hctsa['Awake'][:, i])
snrrhos[:, 1] = multipletests(snrrhos[:, 1], method='bonferroni')[1]
sig = np.where(snrrhos[:, 1] < 0.05)[0]

# regress SNR from these hctsa features
hctsa_reg = hctsa['Awake'].copy()

for i in range(hctsa['Awake'].shape[1]):
    if i in sig:
        hctsa_reg[:, i] = regress_out(hctsa['Awake'][:, i], snr)

# calculate spearman r between hctsa features and type densities
types = [type1l, type1s, type2]

rhos = np.zeros((len(types), hctsa_reg.shape[1], 2))
pvals_corrected = np.zeros((len(types), hctsa_reg.shape[1], 2))
for n, type in enumerate(types):
    print(n)
    for i in range(hctsa_reg.shape[1]):
        rhos[n, i, :] = spearmanr(type, hctsa_reg[:, i], nan_policy='omit')
    pvals_corrected[n, :, 0] = multipletests(rhos[n, :, 1],
                                             method='bonferroni')[1]
np.savez(path+'results/HCTSA/'
         + 'hctsa_norm_zscored_noexcl_snrregressed-corrs_Awake',
         rhos=rhos, pvals_corrected=pvals_corrected)


"""
check framewise displacement
"""

# correlated fd with each mouses' time-series
fd_rhos = np.zeros((len(ts), len(ont_idx)))
for mouse in range(len(ts)):
    for region in range(len(ont_idx)):
        fd_rhos[mouse, region] = spearmanr(fd[mouse, :],
                                           ts[mouse][0][ont_idx[region], :])[0]

# violinplot of corrs for each mouse
plt.figure()
sns.violinplot(data=fd_rhos.T)
plt.xlabel('mouse')
plt.ylabel('spearman r with FD')
plt.savefig(path+'figures/eps/violinplot_fdcorrs.eps')

spearmanr(np.mean(fd_rhos, axis=0), type1l)
spearmanr(np.mean(fd_rhos, axis=0), type2)

"""
Cell types

good news is that everything in synaptome regions is also in cell regions,
so I can use the synaptome acronyms in region_mapping_fc to average cell
density data
"""

# load cell density data
den = pd.read_csv(path+'data/cellatlas_ero2018.csv', index_col=0)

df = dict([])

# for each region in 88-region parc (but unilateral, hence ::2 or 1::2)
regions = region_mapping[lfunc].sort_values(by='ontology').iloc[::2].iterrows()
for index, row in regions:
    # get regions in cell density data that correspond
    # to this region in fc data
    relevant_rows = den[den['acronym'].isin(row['synaptome_acr'])]
    # average across regions
    df[row['fc81_acr']] = relevant_rows.iloc[:, 1:-1].mean(axis=0).values

cells = pd.DataFrame.from_dict(df, orient='index', columns=den.columns[1:-1])

# calculate correlations + corrected p-values
cell_rhos = np.zeros((len(types), cells.shape[1], 2))
for n, type in enumerate(types):
    for i in range(cells.shape[1]):
        cell_rhos[n, i, :] = spearmanr(type[1::2], cells.iloc[:, i])
    cell_rhos[n, :, 1] = multipletests(cell_rhos[n, :, 1],
                                       method='bonferroni')[1]

# plot heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(cell_rhos[:, :, 0].T, square=True, annot=True,
            cmap=cmap_div, vmin=-1, vmax=1, ax=ax, linewidths=.5)
for i in range(cell_rhos.shape[0]):
    for j in range(cell_rhos.shape[1]):
        if cell_rhos[i, j, 1] < 0.05:
            ax.text(i + 0.5, j + 0.5, '*', ha='center', va='center',
                    color='white', fontsize=12)
ax.set_xticklabels(['1l', '1s', '2'])
ax.set_xlabel('synapse type')
ax.set_yticklabels(cells.columns, rotation=0)
fig.savefig(path+'figures/eps/heatmap_celltypes.eps')

# compare cell types with synapse density data
fig, axs = plt.subplots(3, 9, figsize=(30, 10))
for i, t in enumerate([type1l, type1s, type2]):
    for ii, celltype in enumerate(cells.keys()):
        scatter_types(t[::2], zscore(cells[celltype]),
                      ont_names, cmap_ontology, axs[i, ii], ont_names_inv[::2])
        axs[i, ii].set_xlabel('type{}'.format(['1l', '1s', '2'][i]))
        axs[i, ii].set_ylabel(celltype)
        axs[i, ii].set_title('r = ' + str(np.round(cell_rhos[i, ii, 0], 4))
                             + ', p = '
                             + str(np.round(cell_rhos[i, ii, 1], 4)))
        axs[i, ii].set_aspect(1.0/axs[i, ii].get_data_ratio(),
                              adjustable='box')
fig.tight_layout()
fig.savefig(path+'figures/eps/scatter_celltypes.eps')

# calculate spearman r between hctsa features and cell types
rhos = np.zeros((len(cells.columns), hctsa['Awake'].shape[1], 2))
pvals_corrected = np.zeros((len(cells.columns), hctsa['Awake'].shape[1], 2))
for n, key in enumerate(cells.keys()):
    print(n)
    for i in range(hctsa['Awake'].shape[1]):
        rhos[n, i, :] = spearmanr(cells[key], hctsa['Awake'][1::2, i])
    pvals_corrected[n, :, 0] = multipletests(rhos[n, :, 1],
                                             method='bonferroni')[1]

with pd.ExcelWriter(path + 'results/HCTSA/hctsa-norm-zscored-noexcl-hits_'
                    + 'p-bonferroni-corrected_cells.xlsx',
                    engine='openpyxl') as writer:
    sheet_created = False
    for n in range(len(cells.columns)):
        sig = np.where(pvals_corrected[n, :, 0] < 0.05)
        if len(sig[0]) == 0:
            continue
        sigsort = np.argsort(abs(rhos[n, sig, 0].flatten()))
        selected_df = features.iloc[sig[0],
                                    features.columns.isin(['Name',
                                                           'Keywords'])]
        selected_df['Spearmanr'] = rhos[n, sig[0], 0]
        selected_df['p_bonferroni'] = pvals_corrected[n, sig[0], 0]
        selected_df.iloc[sigsort].to_excel(writer,
                                           sheet_name=cells.columns[
                                               n].split(' ')[0],
                                           index=False)
        sheet_created = True
    if not sheet_created:
        print(state + ' ' + str(n))
