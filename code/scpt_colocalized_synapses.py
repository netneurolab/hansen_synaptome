"""
analyze colocalized synapse types
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from palettable.colorbrewer.sequential import (PuBuGn_4,
                                               PuRd_4,
                                               PuBuGn_8)
from netneurotools.plotting import (sort_communities,
                                    _grid_communities)
from scipy.stats import spearmanr


path = '/home/jhansen/gitrepos/hansen_synaptome/'


"""
set-up
"""

synden = pd.read_excel(path + 'data/synaptome/mouse_liu2018/'
                       + 'Type_density_Ricky.xlsx', sheet_name=0, index_col=0)

# synapse regions
synregions = pd.read_pickle(path+'data/synaptome/mouse_liu2018/' +
                            'synregions_info.pkl')

# synapse type indices
type1idx = np.arange(0, 11)
type1idxl = np.array([1, 2, 3, 4, 9, 10])  # except 10 has short lifespan
type1idxs = np.array([0, 5, 6, 7, 8])
type2idx = np.arange(11, 18)
type3idx = np.arange(18, 37)

# make diverging colourmap
teals = PuBuGn_4.mpl_colors
teals.reverse()
reds = PuRd_4.mpl_colors
teals.extend(reds)
teals[0] = PuBuGn_8.mpl_colors[-1]
del teals[4]
cmap_div = LinearSegmentedColormap.from_list('cmap', teals, N=256)


"""
plot synapse density matrix
"""

# get synapse type clusters
c3 = np.load(path+'results/synapse_coloc_clusters_c3.npy')
mat = np.corrcoef(synden.values[type3idx, :])  # type2 similarity
inds = sort_communities(mat, c3)

type_order = np.array([1, 2, 4, 9, 3, 10, 8, 6,
                       0, 7, 5])
type_order = np.concatenate((type_order, type2idx, inds + 18))
np.save(path+'data/synaptome/mouse_liu2018/type_order.npy', type_order)
# order of brain regions (by ontology)
region_order = synregions.sort_values(by='major_region').index

border_positions = []
current_region = None
for idx, region in enumerate(synregions['major_region'].values[region_order]):
    if region != current_region:
        border_positions.append(idx)
        current_region = region

# synapse density matrix
fig, axs = plt.subplots(1, 2, figsize=(24, 8))
sns.heatmap(synden.values[np.ix_(type_order, region_order)], ax=axs[0],
            cmap=cmap_div, xticklabels=False, vmin=0, vmax=1, rasterized=True)
axs[0].set_xticks(border_positions)
axs[0].set_xticklabels(synregions.loc[region_order, :][
    'major_region'].cat.categories, rotation=45)
axs[0].set_xlabel('regions')
axs[0].set_ylabel('synapse type')

# synapse similarity matrix
sns.heatmap(np.corrcoef(synden.values[type_order, :]),
            ax=axs[1], cmap=cmap_div, vmin=-1, vmax=1, square=True,
            linewidths=.5, xticklabels=False, yticklabels=False)
# make borders for similarity matrix
communities = np.concatenate([
    np.full(len(type1idxl), 1),
    np.full(len(type1idxs), 2),
    np.full(len(type2idx), 3),
    np.full(np.sum(c3 == 1), 4),
    np.full(np.sum(c3 == 2), 5),
    np.full(np.sum(c3 == 3), 6)
])
bounds = _grid_communities(communities)
bounds[0] += 0.2
bounds[-1] -= 0.2
for n, edge in enumerate(np.diff(bounds)):
    axs[1].add_patch(patches.Rectangle((bounds[n], bounds[n]),
                                       edge, edge, fill=False, linewidth=2,
                                       edgecolor='k'))

fig.tight_layout()
fig.savefig(path+'figures/eps/heatmap_synden_allsynapses.eps')

""""
correlate type-3 subtypes with type 1 and 2
"""

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(np.mean(synden.values[type1idxl, :], axis=0),
               np.mean(synden.values[type3idx[c3 == 2], :], axis=0),
               s=10)
axs[0].set_xlabel('long-lifetime PSD95 density')
axs[0].set_ylabel('colocalized density (cluster 2)')
r, p = spearmanr(
    np.mean(synden.values[type1idxl, :], axis=0),
    np.mean(synden.values[type3idx[c3 == 2], :], axis=0)
)
axs[0].set_title('r = {:.2f}, p = {:.2e}'.format(r, p))

axs[1].scatter(np.mean(synden.values[type2idx, :], axis=0),
               np.mean(synden.values[type3idx[c3 == 1], :], axis=0),
               s=10)
axs[1].set_xlabel('SAP102 density')
axs[1].set_ylabel('colocalized density (cluster 1)')
r, p = spearmanr(
    np.mean(synden.values[type2idx, :], axis=0),
    np.mean(synden.values[type3idx[c3 == 1], :], axis=0)
)
axs[1].set_title('r = {:.2f}, p = {:.2e}'.format(r, p))
fig.suptitle('all 775 regions')
fig.savefig(path+'figures/eps/scatter_coloccorrs_all775reg.eps')

# now repeat for 137 regions
(type1, type1l, type1s, type2,
 type3, type3c1, type3c2) = np.load(path
                                    + 'data/synaptome/mouse_liu2018/'
                                    + 'type_densities_137.npz'
                                    ).values()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(type1l, type3c2)
axs[0].set_xlabel('long-lifetime PSD95 density')
axs[0].set_ylabel('colocalized density (cluster 2)')
r, p = spearmanr(type1l, type3c2)
axs[0].set_title('r = {:.2f}, p = {:.2e}'.format(r, p))

axs[1].scatter(type2, type3c1)
axs[1].set_xlabel('SAP102 density')
axs[1].set_ylabel('colocalized density (cluster 1)')
r, p = spearmanr(type2, type3c1)
axs[1].set_title('r = {:.2f}, p = {:.2e}'.format(r, p))
fig.suptitle('sc parc (137 regions)')
fig.savefig(path+'figures/eps/scatter_coloccorrs_sc137.eps')

# now repeat for 88 regions
(type1, type1l, type1s, type2,
 type3, type3c1, type3c2) = np.load(path
                                    + 'data/synaptome/mouse_liu2018/'
                                    + 'type_densities_88.npz'
                                    ).values()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(type1l, type3c2)
axs[0].set_xlabel('long-lifetime PSD95 density')
axs[0].set_ylabel('colocalized density (cluster 2)')
r, p = spearmanr(type1l, type3c2)
axs[0].set_title('r = {:.2f}, p = {:.2e}'.format(r, p))

axs[1].scatter(type2, type3c1)
axs[1].set_xlabel('SAP102 density')
axs[1].set_ylabel('colocalized density (cluster 1)')
r, p = spearmanr(type2, type3c1)
axs[1].set_title('r = {:.2f}, p = {:.2e}'.format(r, p))
fig.suptitle('sc parc (88 regions)')
fig.savefig(path+'figures/eps/scatter_coloccorrs_fc88.eps')
