"""
sc-fc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, wilcoxon
from scipy.sparse.linalg import expm
import scipy.io as spio
import mat73
from ast import literal_eval
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import squareform, pdist
import pickle


def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * \
        (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared, SS_Residual


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


def scatter_types(x, y, ont_names, cmap_ontology, ax, rpvals=None):
    for idx, (name, colour) in enumerate(zip(ont_names,
                                             cmap_ontology)):
        mask = ont_names_inv == idx
        if rpvals is None:
            ax.scatter(x=x[mask], y=y[mask], label=name, color=colour)
        else:
            ax.scatter(x=x[mask], y=y[mask], label=name, color=colour,
                       linewidths=(rpvals[mask] < 0.05) * 1, edgecolors='k')


path = "/home/jhansen/gitrepos/hansen_synaptome/"

"""
load
"""

# SC
sc = loadmat(path+'data/structure/sc_oh2012.mat')

# fMRI time-series
ts = mat73.loadmat(path+'data/function/' +
                   'Gozzi/BOLD_timeseries_Awake.mat')['BOLD_timeseries_Awake']
ts_halo = mat73.loadmat(path+'data/function/Gozzi/'
                        + 'BOLD_timeseries_Halo.mat')['BOLD_timeseries_Halo']
ts_med = mat73.loadmat(path+'data/function/Gozzi/'
                       + 'BOLD_timeseries_MedIso.mat'
                       )['BOLD_timeseries_MedIso']

# fc regions for ontology
fcregions = pd.read_excel(path+'data/function/Gozzi/' +
                          'rois_id_acr_names_N_182_ORDER_and_Exclusions.xlsx',
                          sheet_name="Exclusions")
fcregions = fcregions[fcregions['REMOVED?'] != 1]
fcregions.reset_index(drop=True, inplace=True)
fcregions = pd.concat([fcregions, fcregions], ignore_index=True)

# region mappings between sc/fc and synaptome
region_mapping_fc = pd.read_csv(path+'data/region_mapping_fc.csv', index_col=0,
                                converters={'synaptome_acr': literal_eval,
                                            'synaptome_idx': literal_eval})
lfunc_fc = region_mapping_fc['synaptome_acr'].apply(lambda x: len(x) != 0)
region_mapping_sc = pd.read_csv(path+'data/region_mapping_sc.csv', index_col=0,
                                converters={'synaptome_acr': literal_eval,
                                            'synaptome_idx': literal_eval})
lfunc_sc = region_mapping_sc['synaptome_acr'].apply(lambda x: len(x) != 0)

intersection_IDs = np.intersect1d(region_mapping_fc[lfunc_fc]['region_id'],
                                  region_mapping_sc[lfunc_sc]['region_id'])

nnodes = len(intersection_IDs)

"""
remap data to common space
"""

# get SC

sc_reorder_idx = region_mapping_sc[region_mapping_sc['region_id'].isin(
    intersection_IDs)].sort_values(by='ontology').index.tolist()

sct = sc['a'].copy()[:, :, [0, 2]]
sct = sct * (sc['a'][:, :, [1, 3]] < 0.05)
sct_rmp = sct[:, :, 0][np.ix_(sc_reorder_idx, sc_reorder_idx)]

# get FC
fc_rh = region_mapping_fc.loc[:80, :]
fc_reorder_idx = fc_rh[fc_rh['region_id'].isin(
    intersection_IDs)].sort_values(by='ontology').index.tolist()

fc = dict([])

for t, name in zip([ts, ts_halo, ts_med], ['awake', 'halo', 'mediso']):
    fc[name] = np.zeros((nnodes, nnodes, len(t)))
    for i, subjts in enumerate(t):
        fc[name][:, :, i] = np.corrcoef(subjts[0][fc_reorder_idx, :])

# get synaptome
syn = np.load(path+'results/synaptome_fc88.npz')
syn_reorder_idx = [region_mapping_fc[lfunc_fc].sort_values(
    by='ontology').index.get_loc(idx) for idx in fc_reorder_idx]

# if using type density instead of synpase similarity
syn = np.load(path + 'data/synaptome/mouse_liu2018/type_densities_88.npz')

# get ontology names
(ont_names, ont_names_idx,
 ont_names_inv) = np.unique(fcregions['MACRO'].values[fc_reorder_idx],
                            return_index=True,
                            return_inverse=True)

# colourmap for scatterplots where points are regions
cmap_ontology = np.array([[0.39607843, 0.76862745, 0.82352941, 1.0],
                          [0.36470588, 0.63137255, 0.69019608, 1.0],
                          [0.32549020, 0.47843137, 0.65098039, 1.0],
                          [0.52941176, 0.54117647, 0.68235294, 1.0],
                          [0.76862745, 0.88627451, 0.73725490, 1.0]
                          ])

"""
structure-function coupling
"""

# calculate communicability using in-degree
row_sum = np.sum(sct_rmp, axis=0)
neg_sqrt = np.power(row_sum, -0.5)
square_sqrt = np.diag(neg_sqrt)
for_expm = square_sqrt @ sct_rmp @ square_sqrt
# calculate matrix exponential of normalized matrix
cmc = expm(for_expm)
cmc[np.diag_indices_from(cmc)] = 0
nnull = 1000
eu = squareform(pdist(region_mapping_fc.loc[fc_reorder_idx, :][
    ['x', 'y', 'z']].values))
euinv = eu.astype('float64')
np.fill_diagonal(euinv, 1)
euinv **= -1

rsq_sc = dict([])
rsq_syn = dict([])

fig, axs = plt.subplots(3, 3, figsize=(13, 15), sharex='row', sharey='row')

for m, state in enumerate(['awake', 'halo', 'mediso']):
    for j, synfeat in enumerate(['type1l', 'type1s', 'type2']):
        rsq_sc[state + '-' + synfeat] = np.zeros([nnodes, ])
        rsq_syn[state + '-' + synfeat] = np.zeros([nnodes, ])
        rnull = np.zeros([nnodes, nnull])

        for i in range(nnodes):
            print('state:', state, ', synapse type:', synfeat, ', node:', i)
            if i == 28:  # not connected
                rsq_sc[state + '-' + synfeat][i] = np.nan
                rsq_syn[state + '-' + synfeat][i] = np.nan
                continue
            y = np.mean(fc[state], axis=2)[:, i]
            x1 = cmc[:, i]
            x2 = syn[synfeat][syn_reorder_idx]
            # x3 = syn['type3c1'][syn_reorder_idx]
            # x4 = syn['type3c2'][syn_reorder_idx]

            x_sc = zscore(x1).reshape(-1, 1)
            x_syn = zscore(np.stack((x1, x2), axis=1)) # add x3 and x4 for coloc version (Fig S10)
            rsq_sc[state + '-' + synfeat][i], res_sc = get_reg_r_sq(x_sc, y)
            rsq_syn[state + '-' + synfeat][i], res_r = get_reg_r_sq(x_syn, y)

        scatter_types(rsq_sc[state + '-' + synfeat],
                      rsq_syn[state + '-' + synfeat],
                      ont_names, cmap_ontology, axs[m, j], rpvals=None)
        axs[m, j].plot(rsq_sc[state + '-' + synfeat],
                       rsq_sc[state + '-' + synfeat], 'k-', linewidth=.5)
        axs[m, j].set_xlabel('Rsq from SC only')
        axs[m, j].set_ylabel('Rsq from SC + ' + synfeat)
        axs[m, j].set_title(state)
fig.tight_layout()
fig.savefig(path+'figures/eps/scatter_scfc_coupling_typedensity.eps')

with open(path+'results/rsq_syn.pkl', 'wb') as file:
    pickle.dump(rsq_syn, file)
with open(path+'results/rsq_sc.pkl', 'wb') as file:
    pickle.dump(rsq_sc, file)

"""
compare distributions
"""

state = 'halo'
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for j, synfeat in enumerate(['type1l', 'type1s', 'type2']):
    x = rsq_syn['awake-' + synfeat] - rsq_sc['awake-' + synfeat]
    y = rsq_syn[state + '-' + synfeat] - rsq_sc[state + '-' + synfeat]
    sns.violinplot([x, y], ax=axs[j])
    t, p = wilcoxon(x, y, nan_policy='omit')
    axs[j].set_title(synfeat + ': p=' + str(p))
    # axs[j].legend(['awake', state])
    axs[j].set_xlabel('Rsq difference')
fig.tight_layout()
fig.savefig(path+'figures/eps/violin_scfc_rsqdiff_{}.eps'.format(state))

# coletta networks
coletta = fcregions["COLETTA NETWORK"][fc_reorder_idx]
networks = networks = ["DMN-MID", "DMN-PLN", "LCN", "OF-BF"]

state = 'halo'
fig, axs = plt.subplots(3, len(networks), figsize=(15, 5))
for j, synfeat in enumerate(['type1l', 'type1s', 'type2']):
    for i, net in enumerate(networks):
        idx = coletta == net
        x = rsq_syn['awake-' + synfeat] - rsq_sc['awake-' + synfeat]
        y = rsq_syn[state + '-' + synfeat] - rsq_sc[state + '-' + synfeat]
        x = x[idx]
        y = y[idx]
        sns.violinplot([x, y], ax=axs[j, i])
        t, p = wilcoxon(x, y, nan_policy='omit')
        axs[j, i].set_title(synfeat + ': p=' + str(p))
        # axs[j].legend(['awake', state])
        axs[j, i].set_xlabel('Rsq difference')
fig.tight_layout()


"""
plot mouse brain (requires separate environment)
"""

# import brainglobe_heatmap as bgh

# with open(path+'results/rsq_syn.pkl', 'rb') as file:
#     rsq_syn = pickle.load(file)
# with open(path+'results/rsq_sc.pkl', 'rb') as file:
#     rsq_sc = pickle.load(file)

# vmin = 0
# vmax = 0
# for key in rsq_sc.keys():
#     vmin = min(vmin, min(rsq_syn[key] - rsq_sc[key]))
#     vmax = max(vmax, max(rsq_syn[key] - rsq_sc[key]))

# for tname in ['type1l', 'type1s', 'type2']:
#     for state in ['awake', 'halo', 'mediso']:
#         print(tname, state)

#         x = rsq_syn[state + '-' + tname] - rsq_sc[state + '-' + tname]
#         x[28] = np.nan

#         data = dict(zip(region_mapping_sc[
#             region_mapping_sc['region_id'].isin(intersection_IDs)].sort_values(
#             by='ontology')['sc213_acr'], x))

#         for orien in ['frontal']:
#             f = bgh.Heatmap(
#                     data,
#                     position=7150,
#                     orientation=orien,
#                     hemisphere=None,
#                     title="rsq diff: {}-{}".format(state, tname),
#                     cmap=PuBuGn_9.mpl_colormap,
#                     vmin=vmin,
#                     vmax=0.25,
#                     format="2D"
#                 ).show(filename=path+'figures/eps/mouse_plots/'
#                        + 'bgh_rsqdiff_{}_{}_{}_fc35.eps'.format(
#                         tname, state, orien),
#                        cbar_label='ontology')
