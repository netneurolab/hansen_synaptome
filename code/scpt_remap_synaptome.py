"""
remap synaptome to SC and FC regions
"""

import numpy as np
import pandas as pd
from abagen.mouse import (get_structure_coordinates,
                          fetch_allenref_structures)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from palettable.colorbrewer.sequential import (PuBuGn_4,
                                               PuRd_4,
                                               PuBuGn_8)
from abagen.mouse import get_structure_info


def make_region_mapping(filepath, col1name, col2name, outpath):
    """
    make region mapping dataframe from acronyms in `filepath` col1 to
    acronyms in `filepath` col2name.
    Importantly this assumes you're getting both hemispheres so it
    duplicates the region mapping in `filepath` (use for fc not sc)
    """

    # load (manually made) region mapping
    region_mapping = {col1name: [],
                      col2name: []}
    with open(filepath, 'r') as file:
        for line in file:
            words = line.strip().split()
            column1 = words[0]
            columns2 = words[1:]
            region_mapping[col1name].append(column1)
            region_mapping[col2name].append(columns2)
    region_mapping = pd.DataFrame(region_mapping)

    # get the ontology order value (from ara_regions),
    # and structural ID
    ontology, region_id = ([] for _ in range(2))
    coords = []  # coords take a while :-(
    for acr in region_mapping[col1name].values:
        reginfo = ara_regions.query("acronym == @acr")
        if len(reginfo) == 0:  # if region (eg VISli) not in ara_regions
            ontology.append(np.nan)
            region_id.append(np.nan)
            data = {'structure_id': np.nan,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan}
            coords.append(pd.DataFrame(data, index=[0]))
            continue
        ontology.append(reginfo['ontology_order'].values[0])
        region_id.append(reginfo['id'].values[0])
        try:
            coords.append(get_structure_coordinates(id=reginfo['id'],
                                                    acronym=acr))
        except ValueError:
            # for SUBd and SUBv get coord for SUB
            if 'SUB' in acr:
                # but reindex SUB to SUBd or SUBv (yikes sorry)
                c = get_structure_coordinates(id=502, acronym='SUB')
                c['structure_id'] = reginfo['id'].values[0]
                coords.append(c)
            else:
                # shouldn't happen and will break the concat if it does
                coords.append(np.nan)
    coords = pd.concat(coords, axis=0, ignore_index=True)
    coords.drop(columns='structure_id', inplace=True)
    coords = pd.concat([coords, coords], ignore_index=True)
    coords.loc[len(coords) / 2:, 'z'] = -coords.loc[len(coords) / 2:, 'z']\
        + 12000

    # duplicate region_mapping because R and L are symmetric
    region_mapping = pd.concat([region_mapping, region_mapping],
                               ignore_index=True)

    region_mapping.insert(2, "ontology", ontology + ontology)  # duplicate RL
    region_mapping.insert(3, "region_id", region_id + region_id)  # same ^
    region_mapping = region_mapping.join(coords)

    # get RH and LH indices separately; stack at the end
    # this results in a list of lists, where each sublist represents the
    # indices of synaptome that are related to a single synaptome region.
    # The number of lists in the list is equal to the length of
    # region_mapping[synaptome_acr]
    synaptome_idx = []
    for i, sublist in enumerate(region_mapping['synaptome_acr'].values):
        if len(sublist) == 0:
            synaptome_idx.append([])
        else:
            idx = []
            for reg in sublist:

                if ',' in reg:  # required due to CUL4,5
                    reg = reg.split(',')[0] + ', ' + reg.split(',')[1]

                if i < len(region_mapping) / 2:  # RH
                    condition = synregions.query("acronym\
                                                 == @reg")['Region_list'].\
                        str.contains('right', case=False)
                else:  # LH
                    condition = synregions.query("acronym\
                                                 == @reg")['Region_list'].\
                        str.contains('left', case=False)
                idx.append(condition[condition].index.to_list())

            synaptome_idx.append(idx)

    region_mapping.insert(2, "synaptome_idx", synaptome_idx)

    if outpath is not None:
        region_mapping.to_csv(outpath)

    return region_mapping


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


def get_synaptome_acronyms(synregions):
    """
    structures in synaptome that aren't in ARA and so don't get counted:
        - Mop (MOp is counted), CENT1 (CENT2 and 3 are counted),
        - DCOmo (DCO is counted), ZIV and ZID (ZI is counted),
    special case decisions:
        - "Layer 6" in synaptome becomes "Layer 6a" instead of 6b
        - "VPPC" in synaptome becomes "VPMpc" instead of VPLpc
        - "lMD" in synaptome becomes "IMD", I think it was a typo
        - "LG" in synaptome becomes "LGd"
        - "AON1l" and "AON1m" become "AON1"
    """

    # get synapse region IDs
    s = fetch_allenref_structures()
    synreginfo = {'acronym': [],
                  'ara_id': []}
    for synreg in synregions['Region_list']:
        strsplit = synreg.split('_')
        if "Layer" in synreg:
            # get acronym depending on if "layer" is at the start or end of str
            if "Layer" in strsplit[-1]:
                acr = strsplit[2].split(' ')[0]
            else:
                acr = strsplit[-1].split(' ')[0]
            # adjust layer number
            if 'Layer2-3' in synreg:
                acr = acr + '2/3'
            elif 'Layer6' in synreg:
                acr = acr + '6a'
            else:
                acr = acr + synreg.split("Layer")[-1][:1]  # get layer number
        else:
            acr = strsplit[-1].split(' ')[0]
            # weird special cases
            if "isl" in synreg:
                acr = "isl"  # remove number
            elif "," in acr:
                acr = acr.split(',')[0] + ', ' + acr.split(',')[1]
            elif acr == "VPPC":
                acr = "VPMpc"
            elif acr == "lMD":
                acr = "IMD"  # lMD is a typo?
            elif acr == "LG":
                acr = "LGd"
            elif "AON1" in acr:
                acr = "AON1"  # remove l/m
        try:
            synid = s.query("acronym == @acr")['id'].values[0]
            synreginfo['ara_id'].append(synid)
            synreginfo['acronym'].append(acr)
        except IndexError:
            print(synreg)
            synreginfo['ara_id'].append(np.nan)
            synreginfo['acronym'].append(np.nan)
            continue

    return pd.concat([synregions, pd.DataFrame(synreginfo)],
                     ignore_index=False, axis=1)


path = '/home/jhansen/projects/proj_synaptome/'


"""
load regions
"""

ara_regions = pd.read_excel(path+'data/structure/regions.xlsx',
                            sheet_name="voxel_count_all ARA")
struct_info = pd.read_csv(path + 'data/gene_expression/'
                          + 'aba_structure_info_query.csv')
synregions = pd.read_excel(path + 'data/synaptome/mouse_liu2018/'
                           + 'Type_density_Ricky.xlsx', sheet_name=1)
synregions = get_synaptome_acronyms(synregions)
scregions = pd.read_excel(path+'data/structure/regions.xlsx',
                          sheet_name="Voxel Count_295 Structures",
                          usecols=['ID', 'Ontology_order', 'Acronym', 'Name',
                                   'Major_Region'])
fcregions = pd.read_excel(path+'data/function/Gozzi/' +
                          'rois_id_acr_names_N_182_ORDER_and_Exclusions.xlsx',
                          sheet_name="Exclusions")
fcregions = fcregions[fcregions['REMOVED?'] != 1]
fcregions.reset_index(drop=True, inplace=True)

# get structure_id_path of major regions
major_region_acr = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL',
                    'TH', 'HY', 'MB', 'P', 'MY', 'CBX', 'CBN']
acr_to_id_dict = {acr: struct_info.query(
    f"acronym == '{acr}'")['id'].values.tolist()
    for acr in major_region_acr}
cmap_ontology = np.array([[0.97647059, 0.88627451, 0.93333333, 1.0],
                          [0.97647059, 0.88627451, 0.93333333, 1.0],
                          [0.39607843, 0.76862745, 0.82352941, 1.0],
                          [0.36470588, 0.63137255, 0.69019608, 1.0],
                          [0.92549020, 0.67058824, 0.80392157, 1.0],
                          [0.32549020, 0.47843137, 0.65098039, 1.0],
                          [0.76862745, 0.65882353, 0.81568627, 1.0],
                          [0.75686275, 0.78039216, 0.89803922, 1.0],
                          [0.52941176, 0.54117647, 0.68235294, 1.0],
                          [0.75686275, 0.78039216, 0.89803922, 1.0],
                          [0.51176471, 0.80392157, 0.73333333, 1.0],
                          [0.76862745, 0.89803922, 0.95294118, 1.0],
                          [0.76862745, 0.88627451, 0.73725490, 1.0]
                          ])

major_region = [""] * len(synregions)

for idx, araid in enumerate(synregions['ara_id']):
    if np.isnan(araid):
        # special cases; I checked these manually
        if 'Isocortex' in synregions['Region_list'][idx]:
            major_region[idx] = 'Isocortex'
        elif 'Hypothalamus' in synregions['Region_list'][idx]:
            major_region[idx] = 'HY'
        elif 'Cerebellum' in synregions['Region_list'][idx]:
            major_region[idx] = 'CBX'
        elif 'Hindbrain' in synregions['Region_list'][idx]:
            major_region[idx] = 'MY'
    else:
        try:
            # Find the matching acronym by checking if the araid exists in
            # any of the id lists in acr_to_id_dict
            major_region[idx] = next(
                (acr for acr, id_list in acr_to_id_dict.items()
                 if any(f'/{id}/' in struct_info.query(
                     "id == @araid")['structure_id_path'].values[0]
                        for id in id_list)), None)
        except TypeError:
            # Handle any special cases like araid == 599.0
            if araid == 599.0:
                major_region[idx] = 'TH'  # Special case for id 599
synregions['major_region'] = pd.Categorical(major_region,
                                            categories=major_region_acr,
                                            ordered=True)

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
load synaptome data
"""

synparamsim = pd.read_excel(path + 'data/synaptome/mouse_liu2018/'
                            + 'synaptome_similarity.xlsx',
                            header=1, index_col=1).iloc[:, 1:]
# LC left is all 0
synden = pd.read_excel(path + 'data/synaptome/mouse_liu2018/'
                       + 'Type_density_Ricky.xlsx', sheet_name=0, index_col=0)

# synaptome lifespan (wholebrain)
lifespan_coarse = pd.read_excel(path+'data/synaptome/mouse_bulovaite2022/'
                                + 'percentage_remaining_regions.xlsx',
                                skiprows=1)
vals = lifespan_coarse.query("Region_name == 'whole brain'").values[0][2:]

# plot synaptome data
# order of synapse types (by lifespan clusters)
type_order = np.array([1, 2, 4, 9, 3, 10, 8, 6,
                       0, 7, 5, 11, 12, 13, 14, 15, 16, 17])
# order of brain regions (by ontology)
region_order = synregions.sort_values(by='major_region').index

border_positions = []
current_region = None
for idx, region in enumerate(synregions['major_region'].values[region_order]):
    if region != current_region:
        border_positions.append(idx)
        current_region = region

fig, axs = plt.subplots(1, 2, figsize=(24, 6))
sns.heatmap(synden.values[np.ix_(type_order, region_order)], ax=axs[0],
            cmap=cmap_div, xticklabels=False, vmin=0, vmax=1, rasterized=True)
axs[0].set_xticks(border_positions)
axs[0].set_xticklabels(synregions.loc[region_order, :][
    'major_region'].cat.categories, rotation=45)
axs[0].set_xlabel('regions')
axs[0].set_ylabel('synapse type')
sns.heatmap(np.corrcoef(synden.values[type_order, :]),
            ax=axs[1], cmap=cmap_div, vmin=-1, vmax=1, square=True,
            linewidths=.5, xticklabels=False, yticklabels=False)
fig.tight_layout()
fig.savefig(path+'figures/eps/heatmap_synden.eps')

# plot whole-brain synapse lifespan
fig, ax = plt.subplots()
sns.barplot(x=list(range(11)), y=vals[type_order[:11]])
ax.set_xticklabels(type_order[:11])
ax.set_ylabel('lifespan')
fig.tight_layout()
fig.savefig(path+'figures/eps/bar_synapselifespan.eps')

"""
Remap synaptome to regions in FC dataset (88 bilateral)
"""

region_mapping = make_region_mapping(filepath=path + 'data/'
                                     + 'mouse_synaptome_fc_ara_mapping.txt',
                                     col1name='fc81_acr',
                                     col2name='synaptome_acr',
                                     outpath=path + 'data/'
                                     + 'region_mapping_fc.csv')

# add SNR to region_mapping
snr = pd.read_csv(path + 'data/function/Gozzi/mean_SNR_masked.csv')

region_mapping_first_81 = region_mapping.iloc[:81]
snr_first_81 = snr.iloc[:81]
region_mapping_last_81 = region_mapping.iloc[-81:]
snr_last_81 = snr.iloc[-81:]

merged_first_81 = region_mapping_first_81.merge(snr_first_81[['ACRO', 'SNR']],
                                                left_on='fc81_acr',
                                                right_on='ACRO',
                                                how='left')
merged_last_81 = region_mapping_last_81.merge(snr_last_81[['ACRO', 'SNR']],
                                              left_on='fc81_acr',
                                              right_on='ACRO',
                                              how='left')

region_mapping = pd.concat([merged_first_81, merged_last_81])
region_mapping.drop(columns=['ACRO'], inplace=True)
region_mapping.reset_index(drop=True, inplace=True)
# drop the NaNs but keep the index val because it corresponds to order in ts
region_mapping = region_mapping.dropna(subset=['region_id'])

# flatten inner two lists of region_mapping['synaptome_idx']
# flatten inner two lists of region_mapping['synaptome_idx']
flattened_list = [
    [] if not sublist
    else [item for inner_list in sublist for item in inner_list]
    for sublist in region_mapping['synaptome_idx'].values
]

# reorder flattened list by ontology so remapped synaptomes are in
# order of structural ontology rather than alphabetical
A = [flattened_list[i] for i in region_mapping.reset_index(drop=True).
     sort_values(by='ontology').index]

# make synaptome that averages over all regions in mapping
synden_rmp = remap_synaptome(synden.values.T, A, 'density')
synparamsim_rmp = remap_synaptome(synparamsim.values, A, 'similarity')

np.savez(path+'results/synaptome_fc88.npz',
         synden=synden_rmp,
         synparamsim=synparamsim_rmp)

np.savez(path+'data/synaptome/mouse_liu2018/type_densities_88.npz',
         type1=np.mean(synden_rmp[:, type1idx], axis=1),
         type1l=np.mean(synden_rmp[:, type1idxl], axis=1),
         type1s=np.mean(synden_rmp[:, type1idxs], axis=1),
         type2=np.mean(synden_rmp[:, type2idx], axis=1),
         type3=np.mean(synden_rmp[:, type3idx], axis=1))


"""
Remap synaptome to regions in SC dataset (137 right hemisphere)
"""

# load (manually made) region mapping
region_mapping = {'sc213_acr': [],
                  'synaptome_acr': []}
with open(path+'data/mouse_synaptome_sc_ara_mapping.txt', 'r') as file:
    for line in file:
        words = line.strip().split()
        column1 = words[0]
        columns2 = words[1:]
        region_mapping['sc213_acr'].append(column1)
        region_mapping['synaptome_acr'].append(columns2)
region_mapping = pd.DataFrame(region_mapping)

# get RH indices associated with these acronyms
# this results in a list of lists, where each sublist represents the indices
# of synaptome that are related to a single synaptome region. The number of
# lists in the list is equal to the length of region_mapping[synaptome_acr]
synaptome_idx = []
for sublist in region_mapping['synaptome_acr'].values:
    if len(sublist) == 0:
        synaptome_idx.append([])
    else:
        idx = []
        for reg in sublist:
            if ',' in reg:  # annoying condition to make query work for CUL4,5
                reg = reg.split(',')[0] + ', ' + reg.split(',')[1]
            condition = synregions.query("acronym == @reg")['Region_list'].\
                str.contains('right', case=False)
            idx.append(condition[condition].index.to_list())
        synaptome_idx.append(idx)
region_mapping.insert(2, "synaptome_idx", synaptome_idx)

# aaand lastly get the ontology order value (from ara_regions), major region,
# and structural ID
ontology, major_region, region_id = ([] for _ in range(3))
coords = []  # coords take a while :-(
for acr in region_mapping['sc213_acr'].values:
    reginfo = ara_regions.query("acronym == @acr")
    ontology.append(reginfo['ontology_order'].values[0])
    major_region.append(scregions.query("Ontology_order == @ontology[-1]")
                        ['Major_Region'].values[0])
    region_id.append(reginfo['id'].values[0])
    try:
        coords.append(get_structure_coordinates(id=reginfo['id'],
                                                acronym=acr))
    except ValueError:
        # for SUBd and SUBv get coord for SUB
        if 'SUB' in acr:
            # but reindex SUB to SUBd or SUBv (yikes sorry)
            c = get_structure_coordinates(id=502, acronym='SUB')
            c['structure_id'] = reginfo['id'].values[0]
            coords.append(c)
        else:
            coords.append(np.nan)
coords = pd.concat(coords, axis=0, ignore_index=True)
coords.set_index(coords['structure_id'], inplace=True)
coords.drop(columns='structure_id', inplace=True)
region_mapping.insert(3, "ontology", ontology)
region_mapping.insert(4, "major_region", major_region)
region_mapping.insert(5, "region_id", region_id)
region_mapping = pd.merge(region_mapping, coords, left_on='region_id',
                          right_index=True, how='left')

region_mapping.to_csv(path+'data/region_mapping_sc.csv')

# flatten inner two lists of region_mapping['synaptome_idx']
flattened_list = [
    [] if not sublist
    else [item for inner_list in sublist for item in inner_list]
    for sublist in region_mapping['synaptome_idx'].values
]

# reorder flattened list by ontology so remapped synaptomes are in
# order of structural ontology rather than alphabetical
A = [flattened_list[i] for i in region_mapping.
     sort_values(by='ontology').index]

# make synaptome that averages over all regions in mapping
synden_rmp = remap_synaptome(synden.values.T, A, 'density')
synparamsim_rmp = remap_synaptome(synparamsim.values, A, 'similarity')

np.savez(path+'results/synaptome_sc137.npz',
         synden=synden_rmp,
         synparamsim=synparamsim_rmp)

np.savez(path+'data/synaptome/mouse_liu2018/type_densities_137.npz',
         type1=np.mean(synden_rmp[:, type1idx], axis=1),
         type1l=np.mean(synden_rmp[:, type1idxl], axis=1),
         type1s=np.mean(synden_rmp[:, type1idxs], axis=1),
         type2=np.mean(synden_rmp[:, type2idx], axis=1),
         type3=np.mean(synden_rmp[:, type3idx], axis=1))

"""
Get all unique regions in synaptome (275) for gene expression
"""

den = pd.read_csv(path+'data/gene_expression/'
                  + 'Data_Sheet_2_Cell_Atlas_for_Mouse_Brain.csv')
ara_regions['name'] = ara_regions['name'].str.replace(',', '')
den = den.merge(ara_regions[['name', 'acronym']], left_on='Regions',
                right_on='name', how='left')
den.drop(columns='name', inplace=True)

common_acronyms = set(den['acronym'].dropna()).intersection(
    set(synregions['acronym']))

# Initialize lists to store data for region_mapping dataframe
acronyms = []
synaptome_idxs = []
ontologies = []
region_ids = []
major_regions = []

# ARA path IDs for major regions
major_region_ids = [354, 703, 698, 315, 549, 1089, 1097,
                    313, 477, 528, 803, 771, 519]

# Iterate over each common acronym to get required data
# takes a while because of get_structure_info for major_region :(
for acr in common_acronyms:
    # Find indices in synregions where acronym matches
    # and Region_list contains "right"
    synaptome_idx = synregions[(synregions['acronym'] == acr)
                               & (synregions['Region_list'].str.contains(
                                   'right'))].index.tolist()

    if synaptome_idx:
        # Get ontology and region_id information from ara_regions
        reginfo = ara_regions.query("acronym == @acr")
        if not reginfo.empty:
            ontology = reginfo['ontology_order'].values[0]
            region_id = reginfo['id'].values[0]
            try:
                major_region = major_region_acr[
                    next((i for i, value in enumerate(major_region_ids)
                          if f'/{value}/' in
                          get_structure_info(acronym=acr)[
                              'structure_id_path'][0]), None)]
            except ValueError:  # some ID formats are buggy, need for format
                if '2/3' in acr:
                    acr_fmt = acr.split('2/3')[0]
                if '-' in acr:
                    acr_fmt = acr.split('-')[0]
                major_region = major_region_acr[
                    next((i for i, value in enumerate(major_region_ids)
                          if f'/{value}/' in
                          get_structure_info(acronym=acr_fmt)[
                              'structure_id_path'][0]), None)]

            # Append data to lists
            acronyms.append(acr)
            synaptome_idxs.append(synaptome_idx)
            ontologies.append(ontology)
            region_ids.append(region_id)
            major_regions.append(major_region)

# Create the region_mapping dataframe
region_mapping = pd.DataFrame({
    'acronym': acronyms,
    'synaptome_idx': synaptome_idxs,
    'ontology': ontologies,
    'region_id': region_ids,
    'major_region': major_regions
})

region_mapping.to_csv(path+'data/region_mapping_gexp.csv')

# reorder flattened list by ontology so remapped synaptomes are in
# order of structural ontology rather than alphabetical
A = region_mapping.sort_values(by='ontology')['synaptome_idx'].values

# make synaptome that averages over all regions in mapping
synden_rmp = remap_synaptome(synden.values.T, A, 'density')
synparamsim_rmp = remap_synaptome(synparamsim.values, A, 'similarity')
syndensim_rmp = np.corrcoef(synden_rmp)

np.savez(path+'results/synaptome_aba275.npz',
         synden=synden_rmp,
         synparamsim=synparamsim_rmp,
         syndensim=syndensim_rmp)
