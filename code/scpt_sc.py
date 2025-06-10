"""
SC-synaptome analyses
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from palettable.colorbrewer.sequential import (PuBuGn_9,
                                               PuBuGn_4,
                                               PuRd_4,
                                               PuBuGn_8)
import scipy.io as spio
from scipy.stats import spearmanr
from ast import literal_eval


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def scatter_types(x, y, ont_names, cmap_ontology, ax):
    for idx, (name, colour) in enumerate(zip(ont_names,
                                             cmap_ontology)):
        mask = ont_names_inv == idx
        ax.scatter(x=x[mask], y=y[mask], label=name, color=colour)


def remap_synaptome(synaptome, region_mapping, type='density'):
    """
    synaptome: (N, T) or (N, N) array-like
        region x type density matrix or region x region similarity matrix
    region_mapping: list, length M
        list of lists or arrays of indices in synaptome that correspond to
        output
    type: str, optional
        'density' (default) if `synaptome.shape == (M, T)`
        'similarity' if `synaptome.shape == (M, M)`
    output: synaptome.shape array-like
        output synaptome, reshaped
    """
    # how many regions in the output synaptome, equal to the number of
    # regions in the mapping that have an associated synaptome region
    nodes = [len(x) > 0 for x in region_mapping]
    nnodes = sum(nodes)
    region_mapping = np.array(region_mapping, dtype=object)[nodes]
    if type == 'density':
        output = np.zeros((nnodes, synaptome.shape[1]))
        for n, mapp in enumerate(region_mapping):
            output[n, :] = np.mean(synaptome[mapp, :], axis=0)
    if type == 'similarity':
        output = np.zeros((nnodes, nnodes))
        for i, imapp in enumerate(region_mapping):
            for j, jmapp in enumerate(region_mapping):
                submatrix = synaptome[np.ix_(imapp, jmapp)]
                # ignore values = 1 (diagonal) in the average
                # (makes no apparent difference)
                filtered_values = submatrix[submatrix != 1]
                if len(filtered_values) > 0:
                    output[i, j] = np.mean(filtered_values)
                else:
                    output[i, j] = 1
    return output


path = "/home/jhansen/gitrepos/hansen_synaptome/"

"""
load synaptome
"""

synden = pd.read_excel(path + 'data/synaptome/mouse_liu2018/'
                       + 'Type_density_Ricky.xlsx', sheet_name=0, index_col=0)

# remapped to SC regions
synden_rmp, synparamsim_rmp = np.load(
    path+'results/synaptome_sc137.npz').values()

(type1, type1l, type1s, type2,
 type3, type3c1, type3c2) = np.load(path
                                    + 'data/synaptome/mouse_liu2018/'
                                    + 'type_densities_137.npz'
                                    ).values()
type1idx = np.arange(0, 11)
type1idxl = np.array([1, 2, 3, 4, 9, 10])  # except 10 has short lifespan
type1idxs = np.array([0, 5, 6, 7, 8])
type2idx = np.arange(11, 18)
type3idx = np.arange(18, 37)

# region_mapping
region_mapping = pd.read_csv(path+'data/region_mapping_sc.csv', index_col=0,
                             converters={'synaptome_acr': literal_eval,
                                         'synaptome_idx': literal_eval})

# lambda function for 135/213 regions
lfunc = region_mapping['synaptome_acr'].apply(lambda x: len(x) != 0)

# idx to sort 137 regions by ontology
ont_idx = region_mapping[lfunc].sort_values(by='ontology').index
# idx to plot ontology names (when sorted)
(ont_names, ont_names_idx,
 ont_names_inv) = np.unique(region_mapping['major_region'].values[ont_idx],
                            return_index=True, return_inverse=True)

cmap_ontology = np.array([[0.97647059, 0.88627451, 0.93333333, 1.0],
                          [0.97647059, 0.88627451, 0.93333333, 1.0],
                          [0.39607843, 0.76862745, 0.82352941, 1.0],
                          [0.36470588, 0.63137255, 0.69019608, 1.0],
                          [0.92549020, 0.67058824, 0.80392157, 1.0],
                          [0.32549020, 0.47843137, 0.65098039, 1.0],
                          [0.75686275, 0.78039216, 0.89803922, 1.0],
                          [0.76862745, 0.65882353, 0.81568627, 1.0],
                          [0.52941176, 0.54117647, 0.68235294, 1.0],
                          [0.51176471, 0.80392157, 0.73333333, 1.0],
                          [0.76862745, 0.89803922, 0.95294118, 1.0],
                          [0.76862745, 0.88627451, 0.73725490, 1.0]
                          ])

nnodes = synparamsim_rmp.shape[0]
mask = np.triu(np.ones((nnodes, nnodes)), 1) > 0

# make diverging colourmap
teals = PuBuGn_4.mpl_colors
teals.reverse()
reds = PuRd_4.mpl_colors
teals.extend(reds)
teals[0] = PuBuGn_8.mpl_colors[-1]
del teals[4]
cmap_div = LinearSegmentedColormap.from_list('cmap', teals, N=256)

"""
plot synapse density and similarity
"""

# get synapse type clusters
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
fig.savefig(path+'figures/eps/heatmap_synden137.eps')

"""
load and plot SC
"""

sc = loadmat(path+'data/structure/sc_oh2012.mat')

# thresholded sc (keep regions where p < 0.05)
# (same as fulcher & fornito 2016 pnas)
sct = sc['a'].copy()[:, :, [0, 2]]
sct = sct * (sc['a'][:, :, [1, 3]] < 0.05)
sct_rmp = sct[:, :, 0][np.ix_(ont_idx, ont_idx)]
sct_rmp_bin = sct_rmp.copy()
sct_rmp_bin[sct_rmp_bin != 0] = 1

# plot sc
fig, axs = plt.subplots(figsize=(10, 10))
sns.heatmap(sct_rmp_bin, square=True, vmin=0, vmax=1,
            cmap=PuBuGn_9.mpl_colormap,
            xticklabels=False, yticklabels=False)
axs.set_yticks(ont_names_idx, ont_names)
fig.tight_layout()
fig.savefig(path+'figures/eps/heatmap_sc137.eps')

"""
correlate with degree
"""

fig, axs = plt.subplots(2, 5, figsize=(25, 10), sharey=True)
for i, type in enumerate([type1l, type1s, type2, type3c1, type3c2]):
    for d, dn in enumerate(['in', 'out']):
        scatter_types(type, np.log(np.sum(sct_rmp, axis=d)),
                      ont_names, cmap_ontology, axs[d, i])
        axs[d, i].set_xlabel('type{}'.format(['1L', '1S',
                                              '2', '3c1', '3c2'][i]))
        axs[d, i].set_ylabel('sc weighted {}degree'.format(dn))
        r, p = spearmanr(type, np.sum(sct_rmp, axis=d))
        axs[d, i].set_title('r = ' + str(np.round(r, 4)) + ', p = '
                            + str(np.round(p, 4)))
        axs[d, i].set_aspect(1.0/axs[d, i].get_data_ratio(),
                             adjustable='box')
fig.tight_layout()
fig.savefig(path+'figures/eps/scatter_synden-type_sc-wdeg_log.eps')

"""
correlate with degree in specific cortical layers
"""

# make a flattened list but pick a specific layer
synden_rmp_layer = dict([])
scrmp_layer_idx = dict([])

for layer in ['1', '2/3', '4', '5', '6']:
    # make flattened list
    flattened_list = []

    for i in range(region_mapping.shape[0]):
        # if it's not in sc (empty)
        if region_mapping['synaptome_idx'].loc[i] == []:
            flattened_list.append([])

        # check whether this is in 6-layer cortex
        elif np.logical_and(any('5' in s for s in
                                region_mapping['synaptome_acr'].loc[i]),
                            any('6' in s for s in
                                region_mapping['synaptome_acr'].loc[i])):

            layer_idx = next((index for index, string in
                              enumerate(region_mapping['synaptome_acr'].loc[i])
                              if layer in string), None)
            if layer_idx is not None:
                flattened_list.append(region_mapping['synaptome_idx'
                                                     ].loc[i][layer_idx])
            else:
                flattened_list.append([])

        else:
            flattened_list.append([])

    # reorder flattened list by ontology so remapped synaptomes are in
    # order of structural ontology rather than alphabetical
    A = [flattened_list[i] for i in region_mapping.
         sort_values(by='ontology').index]

    # make synaptome that averages over all regions in mapping
    synden_rmp_layer[layer] = remap_synaptome(synden.values.T, A, 'density')

    # idx to sort 135 regions by ontology
    condition = [bool(sublist) for sublist in flattened_list]
    scrmp_layer_idx[layer] = region_mapping[condition].sort_values(
        by='ontology').index

rhos = np.zeros((5, 10))  # layers x (type x in/out degree)
# correlate each layer's type density with sc weighted degree
for t, typeidx in enumerate([type1idx, type1idxl, type1idxs,
                             type2idx, type3idx]):
    fig, axs = plt.subplots(2, len(synden_rmp_layer.keys()), figsize=(20, 10))
    for i, key in enumerate(synden_rmp_layer.keys()):
        x1 = np.sum(sct[:, :, 0], axis=1)[scrmp_layer_idx[key]]
        x2 = np.sum(sct[:, :, 0], axis=0)[scrmp_layer_idx[key]]
        y = np.mean(synden_rmp_layer[key][:, typeidx], axis=1)
        r1, p1 = spearmanr(x1, y)
        r2, p2 = spearmanr(x2, y)
        rhos[i, 2*t] = r2
        rhos[i, 2*t+1] = r1
        axs[0, i].scatter(x1, y)
        axs[0, i].set_xlabel('sc weighted outdegree')
        axs[0, i].set_ylabel('synapse density')
        axs[0, i].set_title('r = ' + str(np.round(r1, 3)) + ', p = '
                            + str(np.round(p1, 4)))
        axs[1, i].scatter(x2, y)
        axs[1, i].set_xlabel('sc weighted indegree')
        axs[1, i].set_ylabel('synapse density')
        axs[1, i].set_title('r = ' + str(np.round(r2, 3)) + ', p = '
                            + str(np.round(p2, 4)))
    fig.tight_layout()
    fig.savefig(path+'figures/png/scatter_synden-type{}'.format(
        ['1', '1L', '1S', '2', '3'][t]) + '-layer_sc-wdeg.png')

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(rhos, square=True, xticklabels=['i', 'o']*5,
            yticklabels=synden_rmp_layer.keys(), linewidths=0.5,
            cmap=cmap_div, vmin=-np.max(abs(rhos)), vmax=np.max(abs(rhos)))
ax.set_xlabel('type1, type1-long, type1-short, type2, type3')
fig.tight_layout()
fig.savefig(path+'figures/eps/heatmap_typeden_layer_sc-deg_corr.eps')

"""
compare lifespan with degree correlation
"""

# load synapse lifespan data
lifespan = pd.read_excel(path + 'data/synaptome/mouse_bulovaite2022/'
                         + 'percentage_remaining_regions.xlsx',
                         skiprows=1)

lspan = lifespan.query("Region_name == 'whole brain'").values[0][2:][:11]

indegrho = np.array([spearmanr(synden_rmp[:, i],
                               np.log(np.sum(sct_rmp, axis=0)))[0]
                     for i in range(11)])
outdegrho = np.array([spearmanr(synden_rmp[:, i],
                                np.log(np.sum(sct_rmp, axis=1)))[0]
                      for i in range(11)])
c = np.zeros((11, ))
c[type1idxs] = 1

fig, axs = plt.subplots(1, 2, figsize=(10, 10), sharey=True)
axs[0].scatter(indegrho, lspan, c=c)
axs[0].set_xlabel('density-indeg spearman r')
axs[0].set_ylabel('lifespan')
r, p = spearmanr(indegrho, lspan)
axs[0].set_title('r = ' + str(np.round(r, 3))
                 + ', p = ' + str(np.round(p, 4)))
axs[0].set_aspect(1.0/axs[0].get_data_ratio(), adjustable='box')
axs[1].scatter(outdegrho, lspan, c=c)
axs[1].set_xlabel('density-outdeg spearman r')
r, p = spearmanr(outdegrho, lspan)
axs[1].set_title('r = ' + str(np.round(r, 3))
                 + ', p = ' + str(np.round(p, 4)))
axs[1].set_aspect(1.0/axs[1].get_data_ratio(), adjustable='box')
fig.tight_layout()
fig.savefig(path+'figures/eps/scatter_lifespan_degree.eps')
