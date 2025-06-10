"""
FC-synaptome analyses
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from palettable.colorbrewer.sequential import (PuBuGn_4,
                                               PuRd_4,
                                               PuBuGn_8)
from scipy.stats import spearmanr
from ast import literal_eval
import mat73


def scatter_types(x, y, ont_names, cmap_ontology, ax):
    for idx, (name, colour) in enumerate(zip(ont_names,
                                             cmap_ontology)):
        mask = ont_names_inv == idx
        ax.scatter(x=x[mask], y=y[mask], label=name, color=colour)


def get_degrhos(fc, synden_rmp, regions):
    regtoplotidx = [np.where(ont_names == item)[0][0]
                    for item in regions]
    mask = np.isin(ont_names_inv, regtoplotidx).astype(bool)
    rhos = np.array([spearmanr(synden_rmp[mask, i],
                               np.sum(np.mean(fc, axis=2), axis=1)[mask])[0]
                    for i in range(11)])
    return rhos


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

# fMRI time-series
ts = mat73.loadmat(path+'data/function/' +
                   'Gozzi/BOLD_timeseries_Awake.mat')['BOLD_timeseries_Awake']
ts_halo = mat73.loadmat(path+'data/function/Gozzi/'
                        + 'BOLD_timeseries_Halo.mat')['BOLD_timeseries_Halo']
ts_med = mat73.loadmat(path+'data/function/Gozzi/'
                       + 'BOLD_timeseries_MedIso.mat'
                       )['BOLD_timeseries_MedIso']

# synaptome
synden_rmp, synparamsim_rmp = np.load(
    path+'results/synaptome_fc88.npz').values()

# make diverging colourmap
teals = PuBuGn_4.mpl_colors
teals.reverse()
reds = PuRd_4.mpl_colors
teals.extend(reds)
teals[0] = PuBuGn_8.mpl_colors[-1]
del teals[4]
cmap_div = LinearSegmentedColormap.from_list('cmap', teals, N=256)

"""
load
"""

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

(type1, type1l, type1s, type2,
 type3, type3c1, type3c2) = np.load(path
                                    + 'data/synaptome/mouse_liu2018/'
                                    + 'type_densities_88.npz'
                                    ).values()

"""
plot the synaptome matrices
"""

plt.ion()

# plot synaptomes and their correlation
nnodes = len(synden_rmp)
mask = np.triu_indices(nnodes, k=1)

# plot synapse density and type x type correlation matrix
type_order = np.load(path+'data/synaptome/mouse_liu2018/type_order.npy')

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(synden_rmp[:, type_order].T, ax=axs[0], cmap=cmap_div,
            xticklabels=False, yticklabels=type_order, vmin=np.min(synden_rmp),
            vmax=np.max(synden_rmp))
axs[0].set_xlabel('regions')
axs[0].set_ylabel('synapse type')
sns.heatmap(np.corrcoef(synden_rmp[:, type_order].T),
            ax=axs[1], cmap=cmap_div, vmin=-1, vmax=1, square=True,
            linewidths=.5, xticklabels=type_order, yticklabels=type_order)
fig.tight_layout()
fig.savefig(path+'figures/eps/heatmap_synden88.eps')

"""
plot the FC matrices and correlate with synapse similarity
"""

syndensim_rmp = spearmanr(synden_rmp[:, type_order], axis=1)[0]

fc = dict([])

for t, name in zip([ts, ts_halo, ts_med], ['awake', 'halo', 'mediso']):
    fc[name] = np.zeros((nnodes, nnodes, len(t)))
    for i, subjts in enumerate(t):
        fc[name][:, :, i] = np.corrcoef(subjts[0][ont_idx, :])

for key, value in fc.items():
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(np.mean(value, axis=2), square=True, xticklabels=False,
                yticklabels=False, ax=axs[0], cmap=cmap_div,
                vmin=-1, vmax=1)
    axs[0].set_title('FC' + key)
    axs[0].set_yticks(ont_names_idx, ont_names)
    axs[1].scatter(syndensim_rmp[mask], np.mean(value, axis=2)[mask], s=2)
    axs[1].set_xlabel('syndensim_rmp')
    axs[1].set_ylabel('FC' + key)
    axs[1].set_aspect(1.0/axs[1].get_data_ratio(), adjustable='box')
    r, p = spearmanr(syndensim_rmp[mask], np.mean(value, axis=2)[mask])
    axs[1].set_title('r = ' + str(np.round(r, 2))
                     + ', p = ' + str(np.round(p, 2)))
    axs[2].scatter(synparamsim_rmp[mask], np.mean(value, axis=2)[mask], s=2)
    axs[2].set_xlabel('synparamsim_rmp')
    axs[2].set_ylabel('FC' + key)
    axs[2].set_aspect(1.0/axs[2].get_data_ratio(), adjustable='box')
    r, p = spearmanr(synparamsim_rmp[mask], np.mean(value, axis=2)[mask])
    axs[2].set_title('r = ' + str(np.round(r, 2)) + ', p = '
                     + str(np.round(p, 2)))
    plt.savefig(path+'figures/eps/heatmap_fc_{}.eps'.format(key))

"""
correlated with strength (weighted degree/hubs)
"""

for key, value in fc.items():
    fig, axs = plt.subplots(1, 5, figsize=(35, 5), sharey=True)
    for i, t in enumerate([type1l, type1s, type2, type3c1, type3c2]):
        x = t
        y = np.sum(np.mean(value, axis=2), axis=1)
        r, p = spearmanr(x, y)
        scatter_types(x, y, ont_names, cmap_ontology, axs[i])
        axs[i].set_xlabel('type{}'.format(['1L', '1S',
                                           '2', '3c1', '3c2'][i]))
        axs[i].set_ylabel('FC' + key)
        axs[i].set_title('r = ' + str(np.round(r, 3)) + ', p = '
                         + str(np.round(p, 4)))
        axs[i].set_aspect(1.0/axs[i].get_data_ratio(), adjustable='box')
    fig.tight_layout()
    fig.savefig(path+'figures/eps/scatter_fc-{}_types.eps'.format(key))

"""
same as above but select which ontological structures
"""

for key, value in fc.items():
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, type in enumerate([type1l, type1s, type2]):
        for idx, (name, colour) in enumerate(zip(ont_names,
                                                 cmap_ontology)):
            regtoplot = ['CTXsp', 'ISO', 'OLF']
            if name not in regtoplot:
                continue
            mask = ont_names_inv == idx
            axs[i].scatter(x=type[mask],
                           y=np.sum(np.mean(value, axis=2), axis=1)[mask],
                           label=name, color=colour)
        axs[i].set_xlabel('type{}'.format(['1L', '1S', '2'][i]))
        axs[i].set_ylabel('fc {} hubs'.format(key))
        regtoplotidx = [np.where(ont_names == item)[0][0]
                        for item in regtoplot]
        mask = np.isin(ont_names_inv, regtoplotidx).astype(bool)
        r, p = spearmanr(type[mask],
                         np.sum(np.mean(value, axis=2), axis=1)[mask])
        axs[i].set_title('r = ' + str(np.round(r, 2)) + ', p = '
                         + str(np.round(p, 2)))
        axs[i].set_aspect(1.0/axs[i].get_data_ratio(), adjustable='box')
    fig.tight_layout()
    fig.savefig(path + 'figures/eps/'
                + 'scatter_synden-type_fc-hubs_{}_'.format(key)
                + 'reg-ctxsp-iso-olf.eps')

"""
compare lifespan with degree corr
"""

# load synapse lifespan data
lifespan = pd.read_excel(path + 'data/synaptome/mouse_bulovaite2022/'
                         + 'percentage_remaining_regions.xlsx',
                         skiprows=1)

lspan = lifespan.query("Region_name == 'whole brain'").values[0][2:][:11]
rhos = get_degrhos(fc['awake'], synden_rmp,
                   ['CTXsp', 'ISO', 'OLF'])

c = np.zeros((11, ))
c[[0, 5, 6, 7, 8]] = 1

fig, ax = plt.subplots()
ax.scatter(rhos, lspan, c=c)
ax.set_xlabel('density-fcdeg spearman r')
ax.set_ylabel('lifespan')
r, p = spearmanr(rhos, lspan)
ax.set_title('r = ' + str(np.round(r, 3))
             + ', p = ' + str(np.round(p, 4)))
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
fig.tight_layout()
fig.savefig(path+'figures/eps/scatter_lifespan_fcdegree_iso-ctxsp-olf.eps')
