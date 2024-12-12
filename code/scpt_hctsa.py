"""
HCTSA
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


def plot_hctsa_corr(feature, state, type_array, type_name, saveout=True):
    fig, ax = plt.subplots()
    i = features.query("Name == @feature").index[0]
    r, p = spearmanr(type_array, hctsa[state][:, i])
    for idx, (name, colour) in enumerate(zip(ont_names,
                                             cmap_ontology)):
        mask = ont_names_inv == idx
        ax.scatter(x=type_array[mask], y=hctsa[state][:, i][mask],
                   label=name, color=colour)
    ax.set_title('r = ' + str(np.round(r, 4))
                 + ', p = ' + str(np.round(p, 5)))
    ax.set_ylabel(feature)
    ax.set_xlabel(type_name + ' denstiy')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.legend()
    fig.tight_layout()
    if saveout:
        fig.savefig(path + 'figures/png/scatter_hctsacorr_{}_{}_{}.png'.
                    format(state, feature, type_name))
        fig.savefig(path + 'figures/eps/scatter_hctsacorr_{}_{}_{}.eps'.
                    format(state, feature, type_name))


def scatter_types(x, y, ont_names, cmap_ontology, ax):
    for idx, (name, colour) in enumerate(zip(ont_names,
                                             cmap_ontology)):
        mask = ont_names_inv == idx
        ax.scatter(x=x[mask], y=y[mask], label=name, color=colour)


path = "/home/jhansen/gitrepos/hansen_synaptome/"

"""
load
"""

fcregions = pd.read_excel(path+'data/function/Gozzi/' +
                          'rois_id_acr_names_N_182_ORDER_and_Exclusions.xlsx',
                          sheet_name="Exclusions")
fcregions = fcregions[fcregions['REMOVED?'] != 1]
fcregions.reset_index(drop=True, inplace=True)
fcregions = pd.concat([fcregions, fcregions], ignore_index=True)
ts = mat73.loadmat(path+'data/function/Gozzi/BOLD_timeseries_Awake.mat')[
                       'BOLD_timeseries_Awake']

# synaptome
synden_rmp, synparamsim_rmp = np.load(
    path+'results/synaptome_fc88.npz').values()

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
type1, type1l, type1s, type2, type3 = np.load(path
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

nnodes = len(ont_idx)
mask = np.triu_indices(nnodes, k=1)

"""
hctsa load
filename conventions:
'normalized' because hctsa outputs were SRS normalized
'zscored' because time-series were zscored before hctsa
'noexcl' because I included all 10 mice (earlier version was with only 9)
'snrregressed' would mean that hctsa features have all had SNR regressed out
'naivep_bonferroni': spearmanr parametric p-value after bonferroni correction
"""

hctsapath = path + 'data/function/Gozzi/HCTSA/'

hctsa = dict([])
for state in ['Awake', 'Halo', 'MedIso']:
    matrix = mat73.loadmat(hctsapath + 'HCTSA_normalized_zscored_noexcl_{}.mat'
                           .format(state))['sharedTS']
    hctsa[state] = np.mean(matrix[ont_idx, :, :], axis=2)

hctsa_awake = mat73.loadmat(hctsapath +
                            'HCTSA_normalized_zscored_noexcl_Awake.mat'
                            )['sharedTS']
features = pd.read_excel(hctsapath + 'HCTSA_normalized_features_noexcl.xlsx',
                         index_col=0).reset_index()

# plot hctsa matrix
fig, ax = plt.subplots(figsize=(15, 5))
sns.heatmap(hctsa['Awake'], ax=ax,
            cmap=cmap_div,
            vmin=0, vmax=1,
            xticklabels=False, yticklabels=False,
            rasterized=True)
for pos in np.where(np.diff(ont_names_inv))[0]:
    ax.hlines(pos + 0.5, 0, hctsa['Awake'].shape[1],
              colors='black', linewidth=0.5)
fig.tight_layout()
fig.savefig(path+'figures/eps/heatmap_hctsamat.eps')

"""
for every synapse type, univariate correlation with hctsa
"""

types = [type1, type1l, type1s, type2, type3]

# run correlations and save output
for state in ['Awake', 'Halo', 'MedIso']:

    print(state)

    # calculate spearman r between hctsa features and type densities
    rhos = np.zeros((len(types), hctsa[state].shape[1], 2))
    pvals_corrected = np.zeros((len(types), hctsa[state].shape[1], 2))
    for n, type in enumerate(types):
        print(n)
        for i in range(hctsa[state].shape[1]):
            rhos[n, i, :] = spearmanr(type, hctsa[state][:, i])
        pvals_corrected[n, :, 0] = multipletests(rhos[n, :, 1],
                                                 method='bonferroni')[1]
    np.savez(path+'results/HCTSA/'
             + 'hctsa_norm_zscored_noexcl-corrs_{}'.format(state),
             rhos=rhos, pvals_corrected=pvals_corrected)

    # make a spreadsheet
    file = np.load(path+'results/HCTSA/hctsa_norm_zscored_noexcl-corrs_{}.npz'
                   .format(state))
    rhos = file['rhos']
    pvals_corrected = file['pvals_corrected']
    with pd.ExcelWriter(path + 'results/HCTSA/hctsa-norm-zscored-noexcl-hits_'
                        + 'naivep_bonferroni-corrected_{}.xlsx'.format(state),
                        engine='openpyxl') as writer:
        sheet_created = False
        for n in range(len(types)):
            sig = np.where(pvals_corrected[n, :, 0] < 0.05)
            if len(sig[0]) == 0:
                continue
            sigsort = np.argsort(abs(rhos[n, sig, 0].flatten()))
            selected_df = features.iloc[sig[0],
                                        features.columns.isin(['Name',
                                                               'Keywords'])]
            selected_df['Spearmanr'] = rhos[n, sig[0], 0]
            selected_df['p_naive_bonferroni'] = pvals_corrected[n, sig[0], 0]
            selected_df.iloc[sigsort].to_excel(writer,
                                               sheet_name='Type{}'.format(
                                                   ['1', '1l', '1s',
                                                    '2', '3'][n]),
                                               index=False)
            sheet_created = True
        if not sheet_created:
            print(state + ' ' + str(n))

# show selected top feature
plot_hctsa_corr('StatAv10',
                'Awake', type1l, 'type1-long')
plot_hctsa_corr('MF_steps_ahead_arma_3_1_6_rmserr_4',
                'Awake', type1s, 'type1-short')
plot_hctsa_corr('DN_OutlierInclude_n_001_mrmd',
                'Awake', type2, 'type2')


"""
SNR
"""

# compare SNR with synapse density data
snr = region_mapping[lfunc].sort_values(by='ontology')['SNR'].values
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, t in enumerate([type1l, type1s, type2]):
    scatter_types(t, snr,
                  ont_names, cmap_ontology, axs[i])
    axs[i].set_xlabel('type{}'.format(['1l', '1s', '2'][i]))
    axs[i].set_ylabel('SNR')
    r, p = spearmanr(t, snr, nan_policy='omit')
    axs[i].set_title('r = ' + str(np.round(r, 4))
                     + ', p = ' + str(np.round(p, 4)))
    axs[i].set_aspect(1.0/axs[i].get_data_ratio(), adjustable='box')
fig.tight_layout()
fig.savefig(path+'figures/eps/scatter_snr_type.eps')

# regress SNR from hctsa features
hctsa_reg = hctsa['Awake'].copy()
snr[np.isnan(snr)] = np.nanmean(snr)

model = LinearRegression()
for i in range(hctsa['Awake'].shape[1]):
    y = hctsa['Awake'][:, i]
    model.fit(snr.reshape(-1, 1), y)  # Fit the model
    y_pred = model.predict(snr.reshape(-1, 1))  # Predicted values based on SNR
    hctsa_reg[:, i] = y - y_pred  # Subtract predicted values to remove SNR

types = [type1, type1l, type1s, type2, type3]

# calculate spearman r between hctsa features and type densities
rhos = np.zeros((len(types), hctsa_reg.shape[1], 2))
pvals_corrected = np.zeros((len(types), hctsa_reg.shape[1], 2))
for n, type in enumerate(types):
    print(n)
    for i in range(hctsa_reg.shape[1]):
        rhos[n, i, :] = spearmanr(type, hctsa_reg[:, i])
    pvals_corrected[n, :, 0] = multipletests(rhos[n, :, 1],
                                             method='bonferroni')[1]
np.savez(path+'results/HCTSA/'
         + 'hctsa_norm_zscored_noexcl_snrregressed-corrs_Awake',
         rhos=rhos, pvals_corrected=pvals_corrected)


"""
plot hctsa correlation coef a la golia
"""

mat = np.load(path+'results/HCTSA/'
              + 'hctsa_norm_zscored_noexcl-corrs_Awake.npz')
rhos = mat['rhos']
pvals = mat['pvals_corrected']

fig, ax = plt.subplots(1, rhos.shape[0], figsize=(16, 5),
                       sharex=True, sharey=True)
for i in range(rhos.shape[0]):
    rhos_sorted = np.sort(abs(rhos[i, :, 0]))
    pvals_sorted = np.array(pvals[i, :, 0])[np.argsort(abs(rhos[i, :, 0]))]
    sigidx = pvals_sorted < 0.05
    nsigidx = pvals_sorted >= 0.05
    ax[i].scatter(np.arange(len(rhos_sorted))[sigidx], rhos_sorted[sigidx],
                  s=2, c='#007d7c')
    ax[i].scatter(np.arange(len(rhos_sorted))[nsigidx], rhos_sorted[nsigidx],
                  s=2, c='#d8d8d8')
    ax[i].set_xlabel('feature')
    ax[i].set_ylabel('spearmanr')
    ax[i].set_title('type {}'.format(['1', '1l', '1s', '2', '3'][i]))
fig.tight_layout()
fig.savefig(path+'figures/eps/scatter_hctsarhos_Awake.eps')

# plot mouse brain (need to do in separate environment)

# for fname in ['StatAv10']:
#     fidx = features['Name'].tolist().index(fname)
#     vmin = np.min(hctsa['Awake'][:, fidx])
#     vmax = np.max(hctsa['Awake'][:, fidx])
#     for hem in ['left', 'right']:
#         if hem == 'left':
#             data = dict(zip(region_mapping[lfunc].sort_values(
#                 by='ontology')['fc81_acr'][1::2],
#                             hctsa['Awake'][:, fidx][1::2]))
#         else:
#             data = dict(zip(region_mapping[lfunc].sort_values(
#                 by='ontology')['fc81_acr'][0::2],
#                             hctsa['Awake'][:, fidx][0::2]))
#         for orien in ['frontal', 'horizontal']:
#             f = bgh.Heatmap(
#                     data,
#                     position=None,
#                     orientation=orien,
#                     hemisphere=hem,
#                     title="ontology",
#                     cmap=PuBuGn_9.mpl_colormap,
#                     vmin=vmin,
#                     vmax=vmax,
#                     format="2D"
#                 ).show(filename=path+'figures/eps/mouse_plots/'
#                        + 'bgh_{}_{}_{}_fc88.eps'.format(fname, orien, hem))

# type 1 region ts to plot:
mouse_idx = 8
reg = ["PA", "EP", "COA", "AUDv", "SSp-bfd", "VISam"]

fig, ax = plt.subplots(6, 2, figsize=(15, 20),
                       sharex=True, sharey=True)  # time-series
for i, r in enumerate(reg):
    ts_dataR = ts[mouse_idx][0][np.where(
        region_mapping['fc81_acr'] == r)[0][0], :]
    ts_dataL = ts[mouse_idx][0][np.where(
        region_mapping['fc81_acr'] == r)[0][1], :]
    ax[i, 0].plot(zscore(ts_dataR))
    ax[i, 0].set_title(r + ' right')
    ax[i, 1].plot(zscore(ts_dataL))
    ax[i, 1].set_title(r + ' left')
fig.suptitle(f'Mouse {mouse_idx}')
fig.tight_layout()
fig.savefig(path
            + 'figures/eps/plot_type1l_ts_mouse{}.eps'.format(mouse_idx))

# type 2 region ts to plot:
mouse_idx = 1
reg = ["MBmot", "SPF", "CA", "BLA", "LA", "PAA"]

fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey=True)
for i, r in enumerate(reg):
    ts_data = ts[mouse_idx][0][np.where(
        region_mapping['fc81_acr'] == r)[0][0], :]
    ax[i//2, i % 2].plot(zscore(ts_data))
    ax[i//2, i % 2].set_title(r)
fig.suptitle(f'Mouse {mouse_idx}')
fig.tight_layout()
fig.savefig(path+'figures/eps/plot_type2_ts_mouse{}.eps'.format(mouse_idx))

"""
compare features (code modified from golia)
"""

# plt.rcParams.update({'font.size': 8}) for type2 features

# for each synapse type
for stype in range(rhos.shape[0]):
    
    # get features that are significant
    sig = np.where(pvals[stype, :, 0] < 0.05)[0]
    if len(sig) == 0:
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
        start = time.time()
        nclust = num_clusters[clustering]
        clusteringResult = AgglomerativeClustering(n_clusters=nclust,
                                                   linkage='average').fit(
                                                   dataMat.T)
        allLabels[clustering, :] = clusteringResult.labels_
        end = time.time()
        print('\nRunning time = ', end-start, 'seconds!')

    # np.save(path + 'results/HCTSA/hierClust_all.npy', allLabels)

    # plot and save all solutions
    for solutionid in range(allLabels.shape[0]):
        nclusters = len(np.unique(allLabels[solutionid, :]))
        print(nclusters)

        # communities = np.asarray(allLabels[solutionid, :])
        communities = allLabels[solutionid, :].flatten().astype(int)
        if 0 in communities:
            communities = communities + 1

        myplot = plotting.plot_mod_heatmap(dataMat, communities, cmap='magma',
                                           rasterized=True, figsize=(30, 30),
                                           xticklabels=features['Name'].iloc[sig].values)
        plt.tight_layout()

        plt.savefig(path + 'figures/png/heatmap_featuresimilarity_type%s_nclusters%s.png'
                    % (['1', '1l', '1s', '2', '3'][stype], nclusters),
                    bbox_inches='tight')

# or do dendrogram method

for stype in range(rhos.shape[0]):

    # get features that are significant
    sig = np.where(pvals[stype, :, 0] < 0.05)[0]
    if len(sig) == 0:
        continue
    
    # feature matrix of hits
    featMat = hctsa['Awake'][:, sig]
    
    # feature similarity matrix
    dataMat = zscore(featMat)
    dataMat = np.abs(spearmanr(dataMat)[0])

    # setting distance_threshold=0 ensures we compute the full tree (we won't
    # be 'cutting' the tree to get a certain number of clusters)
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,
                                    linkage='average')
    model = model.fit(dataMat.T)

    fig, axis = plt.subplots(1, 1, figsize=(15, 7))
    plt.title('Hierarchical Clustering Dendrogram: Type {}'.format(
        ['1', '1l', '1s', '2', '3'][stype]))
    # plot dendrogram; 'CodeString' corresponds to each feature's hctsa code string
    plot_dendrogram(model, truncate_mode='level', p=0,
                    labels=features['Name'].iloc[sig].values,
                    leaf_rotation=90, ax=axis)
    plt.tight_layout()
    plt.savefig(path + 'figures/png/dendogram_type{}.png'
                .format(['1', '1l', '1s', '2', '3'][stype]),
                bbox_inches='tight', dpi=300)
