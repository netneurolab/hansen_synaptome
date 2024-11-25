"""
plot mouse brains using brainglobe
"""

import numpy as np
import pandas as pd
import brainglobe_heatmap as bgh
from ast import literal_eval
from palettable.colorbrewer.sequential import PuBuGn_9
from matplotlib.colors import LinearSegmentedColormap

path = "/home/jhansen/projects/proj_synaptome/"

# min and max of type densities across both fc and sc parcs
vmint1 = 0.0034
vmaxt1 = 0.8144

"""
SC
"""

type1, type1l, type1s, type2, type3 = np.load(path
                                              + 'data/synaptome/mouse_liu2018/'
                                              + 'type_densities_137.npz'
                                              ).values()

# region_mapping
region_mapping = pd.read_csv(path+'data/region_mapping_sc.csv', index_col=0,
                             converters={'synaptome_acr': literal_eval,
                                         'synaptome_idx': literal_eval})

# lambda function for 135/213 regions
lfunc = region_mapping['synaptome_acr'].apply(lambda x: len(x) != 0)

# idx to sort 135 regions by ontology
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
cmap = LinearSegmentedColormap.from_list('custom', cmap_ontology, N=12)

# plot

# if pos, go for 3000 - 12000

for t, tname in zip([type1l, type1s, type2],
                    ['type1l', 'type1s', 'type2']):
    data = dict(zip(region_mapping[lfunc].sort_values(
        by='ontology')['sc213_acr'], t))
    data['SUB'] = data.pop('SUBd')  # SUBd not in atlas but SUB is
    for orien in ['frontal', 'sagittal', 'horizontal']:
        f = bgh.Heatmap(
            data,
            position=None,
            orientation=orien,
            hemisphere=None,
            title="{} density".format(tname),
            cmap=PuBuGn_9.mpl_colormap,
            vmin=vmint1,
            vmax=vmaxt1,
            format="2D"
        ).show(filename=path+'figures/eps/mouse_plots/'
               + 'bgh_{}_{}_sc137.eps'.format(tname, orien),
               cbar_label='density')

"""
FC
"""

type1, type1l, type1s, type2, type3 = np.load(path
                                              + 'data/synaptome/mouse_liu2018/'
                                              + 'type_densities_88.npz'
                                              ).values()

# region_mapping
region_mapping = pd.read_csv(path+'data/region_mapping_fc.csv', index_col=0,
                             converters={'synaptome_acr': literal_eval,
                                         'synaptome_idx': literal_eval})
# lambda function for 88/162 regions
lfunc = region_mapping['synaptome_acr'].apply(lambda x: len(x) != 0)

fcregions = pd.read_excel(path+'data/function/Gozzi/' +
                          'rois_id_acr_names_N_182_ORDER_and_Exclusions.xlsx',
                          sheet_name="Exclusions")
fcregions = fcregions[fcregions['REMOVED?'] != 1]
fcregions.reset_index(drop=True, inplace=True)
fcregions = pd.concat([fcregions, fcregions], ignore_index=True)
# idx to sort 88 regions by ontology
ont_idx = region_mapping[lfunc].sort_values(by='ontology').index
(ont_names, ont_names_idx,
 ont_names_inv) = np.unique(fcregions['MACRO'].values[ont_idx],
                            return_index=True,
                            return_inverse=True)

# if pos, go from 3000 - 10000

for t, tname in zip([type1l, type1s, type2],
                    ['type1l', 'type1s', 'type2']):
    for hem in ['left', 'right']:
        for orien in ['frontal']:
            if hem == 'left':
                data = dict(zip(region_mapping[lfunc].sort_values(
                    by='ontology')['fc81_acr'][1::2], t[1::2]))
            else:
                data = dict(zip(region_mapping[lfunc].sort_values(
                    by='ontology')['fc81_acr'][0::2], t[0::2]))
            f = bgh.Heatmap(
                    data,
                    position=7750,  # 7750 for fig 2a; None for fig 1
                    orientation=orien,
                    hemisphere=hem,
                    title="{} density".format(tname),
                    cmap=PuBuGn_9.mpl_colormap,
                    vmin=vmint1,
                    vmax=vmaxt1,
                    format="2D"
                ).show(filename=path+'figures/eps/mouse_plots/'
                       + 'bgh_{}_{}_{}_pos7750_fc88.eps'.format(
                        tname, hem, orien),
                       cbar_label='density')

"""
plot mouse brain ontology
"""

data = dict(zip(region_mapping[lfunc].sort_values(
    by='ontology')['sc213_acr'], ont_names_inv.astype(float)))
data['SUB'] = data.pop('SUBd')  # SUBd not in atlas but SUB is

for orien in ['frontal', 'sagittal', 'horizontal']:
    f = bgh.Heatmap(
            data,
            position=None,
            orientation=orien,
            hemisphere=None,
            title="ontology",
            cmap=cmap,
            vmin=min(data.values()),
            vmax=max(data.values()),
            format="2D",
            label_regions=True
        ).show(filename=path+'figures/eps/mouse_plots/'
               + 'bgh_ontology_{}_sc137.eps'.format(orien),
               cbar_label='ontology')
