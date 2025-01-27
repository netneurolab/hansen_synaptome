"""
gene expression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.io import savemat
from statsmodels.stats.multitest import multipletests
from abagen.mouse import (get_unionization_from_gene,
                          fetch_allenref_genes,
                          get_gene_info)
from abagen.correct import normalize_expression
from joblib import Parallel, delayed
from ast import literal_eval
import time


def get_mouse_gene_expression(gi, ids):
    try:
        exp = get_unionization_from_gene(id=genes.loc[gi]['id'],
                                         acronym=genes.loc[gi]['acronym'],
                                         name=genes.loc[gi]['name'],
                                         structures=ids,
                                         slicing_direction='coronal',
                                         verbose=False)
        exp = exp.reset_index(level=[0])  # only keep structure_id in index
        return exp['expression_density'].rename(genes.loc[gi]['acronym'])
    except ValueError:
        return []


def scatter_types(x, y, ont_names, cmap_ontology, ax):
    for idx, (name, colour) in enumerate(zip(ont_names,
                                             cmap_ontology)):
        mask = ont_names_inv == idx
        ax.scatter(x=x[mask], y=y[mask], label=name, color=colour)


path = "/home/jhansen/projects/proj_synaptome/"

"""
load
"""

region_mapping = pd.read_csv(path+'data/region_mapping_gexp.csv',
                             index_col=0,
                             converters={'synaptome_acr': literal_eval,
                                         'synaptome_idx': literal_eval})

# idx to sort 275 regions by ontology
ont_idx = region_mapping.sort_values(by='ontology').index
# idx to plot ontology names (when sorted)
(ont_names, ont_names_idx,
 ont_names_inv) = np.unique(region_mapping['major_region'].values[ont_idx],
                            return_index=True, return_inverse=True)

# cerebellar cortex + nuclei get compressed into "cerebellum"
# medulla (8) and pons (1) get compressed into "hindbrain"
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

# load synapse type densities
type1, type1l, type1s, type2, type3 = np.load(path
                                              + 'data/synaptome/mouse_liu2018/'
                                              + 'type_densities_275.npz'
                                              ).values()

# gene expression
gexp_sag = pd.read_csv(path +
                       'data/gene_expression/abagen_mouse_sagittal_275.csv')
gexp_cor = pd.read_csv(path +
                       'data/gene_expression/abagen_mouse_coronal_275.csv')

"""
get gene expression (take a while - don't run again)
"""

genes = fetch_allenref_genes()
# get acronym names of regions in union of synaptome and cell density (275)
ids = region_mapping['region_id'].values

# Parallelize this shit
# gexp_list = Parallel(n_jobs=40)(delayed(get_mouse_gene_expression)(gi, ids)
#                                 for gi in range(genes.shape[0]))

# or... no parallelization because connection errors >:(
gexp_list = []
for gi in range(genes.shape[0]):
    value_error = False
    if gi % 100 == 0:
        print(gi)
    while True:
        try:
            exp = get_unionization_from_gene(id=genes.loc[gi]['id'],
                                             acronym=genes.loc[gi]['acronym'],
                                             name=genes.loc[gi]['name'],
                                             structures=ids,
                                             slicing_direction='sagittal',
                                             verbose=False)
            break  # Break out of the while loop if successful
        except (ValueError, ConnectionResetError) as e:
            if isinstance(e, ValueError):
                value_error = True
                break  # Skip this gene if a ValueError occurs
            if isinstance(e, ConnectionResetError):
                print(f"ConnectionResetError occurred at gi={gi}."
                      + " Retrying after 5 seconds...")
                time.sleep(5)  # Pause for 5 seconds before retrying
    if value_error:
        continue
    exp = exp.reset_index(level=[0])  # Only keep structure_id in index
    e = exp['expression_density'].rename(genes.loc[gi]['acronym'])
    gexp_list.append(e)

# remove empty lists
gexp_list = [item for item in gexp_list if not isinstance(item, list)]

# get reordering of structure IDs according to ontology order
structure_id_order = region_mapping.\
    sort_values(by='ontology')['region_id'].values

# save out gene expression dataframe (ordered by ontology)
gexp = pd.concat(gexp_list, axis=1).reindex(structure_id_order)
gexp.to_csv(path+'data/gene_expression/abagen_mouse_sagittal_275.csv')

"""
normalize gene expression with SRS
(makes no difference because I'm using spearman...)
"""

gexp_list = [gexp_cor.iloc[:, 1:][[col]]
             for col in gexp_cor.iloc[:, 1:].columns]
gexp_cor_norm = normalize_expression(gexp_list, norm='srs')
gexp_cor_norm = pd.concat(gexp_cor_norm, axis=1)

"""
get robust genes
"""

# get genes in both gexp_cor and gexp_sag that have corr(genes) >= 0.7
gexp_cor_filtered = gexp_cor.drop(columns=['structure_id'])
gexp_sag_filtered = gexp_sag.drop(columns=['structure_id'])

# Finding the common genes (columns) in both dataframes
common_genes = set(gexp_cor_filtered.columns).intersection(
    set(gexp_sag_filtered.columns))

# Initializing an empty list to store genes with correlation >= 0.7
high_corr_genes = []

# Iterating through the common genes and calculating the correlation
for gene in common_genes:
    correlation = gexp_cor_filtered[gene].corr(gexp_sag_filtered[gene])
    if correlation >= 0.7:
        high_corr_genes.append(gene)
np.save(path+'data/gene_expression/high_corr_genes.npy', high_corr_genes)
savemat(path+'data/gene_expression/high_corr_genes.mat',
        {'high_corr_genes': high_corr_genes})

"""
for every synapse type, univariate correlation with gene exp
"""

# load high_corr_genes
high_corr_genes = list(np.load(path
                       + 'data/gene_expression/high_corr_genes.npy'))

# calculate spearman r between gene exp and type densities
types = [type1, type1l, type1s, type2, type3]

# gexp_values = gexp_sag.values[:, 1:]
gexp_values = gexp_cor_norm[high_corr_genes].values
rhos = np.zeros((len(types), gexp_values.shape[1], 2))
pvals_corrected = np.zeros((len(types), gexp_values.shape[1]))

for n, type in enumerate(types):
    print(n)
    out = Parallel(n_jobs=40)(delayed(spearmanr)(type, gexp_values[:, i],
                                                 nan_policy='omit')
                              for i in range(gexp_values.shape[1]))
    rhos[n, :, :] = np.array(out)
    pvals_corrected[n, :] = multipletests(rhos[n, :, 1],
                                          method='bonferroni')[1]

savemat(path+'results/gene_expression/synapsetype_gexp_rhos_corsagunion.mat',
        {'rhos': rhos, 'pvals_corrected': pvals_corrected})

# save out
with pd.ExcelWriter(path+'results/gene_expression/gexphits_corsagunion.xlsx',
                    engine='openpyxl') as writer:
    for n in range(len(types)):
        sig = np.where(pvals_corrected[n, :] < 0.05)
        if len(sig[0]) == 0:
            continue
        sigsort = np.argsort(rhos[n, sig, 0].flatten())

        sorted_rhos = rhos[n, sig, 0].flatten()[sigsort]
        sorted_glabels = np.array(list(
            gexp_cor[high_corr_genes].keys()))[sig][sigsort]
        sorted_pvals = pvals_corrected[n, sig].squeeze()[sigsort]

        df = pd.DataFrame({'Gene': sorted_glabels,
                           'Spearmanr': sorted_rhos,
                           'bonferroni_p': sorted_pvals})
        df.to_excel(writer, sheet_name='Type{}'.format(
            ['1', '1l', '1s', '2', '3'][n]), index=False)

# some selected vignettes
fig, ax = plt.subplots(1, 5, figsize=(25, 5))
for i, g in enumerate(['Dact2', 'Slc9a9', 'Lamp5', 'Kcnq5', 'Agt']):
    scatter_types(type1l, gexp_cor_norm[g], ont_names, cmap_ontology, ax[i])
    ax[i].set_title('r = ' +
                    str(np.round(rhos[1, high_corr_genes.index(g), 0], 4))
                    + ', p = ' +
                    str(np.round(pvals_corrected[1, high_corr_genes.index(g)],
                                 5)))
    ax[i].set_ylabel(g + ' expression')
    ax[i].set_xlabel('Type1_long denstiy')
    ax[i].set_aspect(1.0/ax[i].get_data_ratio(), adjustable='box')
fig.tight_layout()
fig.savefig(path + 'figures/eps/scatter_gexpcorrs_type1long.eps')

fig, ax = plt.subplots(1, 5, figsize=(25, 5))
for i, g in enumerate(['Agt', 'Slc6a3', 'Phactr1', 'Syt12', 'Limch1']):
    scatter_types(type1s, gexp_cor_norm[g], ont_names, cmap_ontology, ax[i])
    ax[i].set_title('r = ' +
                    str(np.round(rhos[2, high_corr_genes.index(g), 0], 4))
                    + ', p = ' +
                    str(np.round(pvals_corrected[2, high_corr_genes.index(g)],
                                 5)))
    ax[i].set_ylabel(g + ' expression')
    ax[i].set_xlabel('Type1_short denstiy')
    ax[i].set_aspect(1.0/ax[i].get_data_ratio(), adjustable='box')
fig.tight_layout()
fig.savefig(path + 'figures/eps/scatter_gexpcorrs_type1short.eps')

fig, ax = plt.subplots(1, 5, figsize=(25, 5))
for i, g in enumerate(['Akap12', 'Kcng4', 'Cacna2d2', 'Kctd9', 'Ddn']):
    scatter_types(type2, gexp_cor_norm[g], ont_names, cmap_ontology, ax[i])
    ax[i].set_title('r = ' +
                    str(np.round(rhos[3, high_corr_genes.index(g), 0], 4))
                    + ', p = ' +
                    str(np.round(pvals_corrected[3, high_corr_genes.index(g)],
                                 5)))
    ax[i].set_ylabel(g + ' expression')
    ax[i].set_xlabel('Type2 denstiy')
    ax[i].set_aspect(1.0/ax[i].get_data_ratio(), adjustable='box')
fig.tight_layout()
fig.savefig(path + 'figures/eps/scatter_gexpcorrs_type2.eps')

"""
Gene Ontology
"""

# get entrez ID for each gene
genes = list(gexp.keys()[1:])
entrezID = np.zeros(len(genes), )

for i, g in enumerate(genes):
    if i % 100 == 0:
        print(i)
    try:
        entrezID[i] = get_gene_info(acronym=g)['entrez_id'].values[0]
    except ValueError:
        print('error: ' + g)
        entrezID[i] = np.nan

geneID = pd.DataFrame({'Gene': genes, 'EntrezID': entrezID})

# save and move to MATLAB
geneID.to_csv(path+'data/gene_expression/abagen_mouse_entrezID.csv',
              index=False)

# plot gene ontology
ncats = 20  # number of categories to plot
for t in ['Type1_long', 'Type1_short', 'Type2']:
    go = pd.read_excel(path + 'results/gene_expression/'
                       + 'categoryScores_median_ngenethresh100_sagittal.xlsx',
                       sheet_name=t)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.scatter(go['categoryScore'][-ncats:], range(ncats),
               edgecolors=None, s=18)
    for i, x in enumerate(go['categoryScore'][-ncats:]):
        ax.hlines(i, 0, x, linestyle='dashed', linewidth=0.5)
    ax.set_xlim([go['categoryScore'][-ncats:].min() - 0.02,
                 go['categoryScore'][-ncats:].max() + 0.01])
    ax.set_yticks(range(ncats))
    ax.set_yticklabels(go['GO_Name'][-ncats:])
    ax.set_xlabel('category score')
    ax.set_title(t)
    fig.tight_layout()
    fig.savefig(path + 'figures/eps/scatter_categoryScores_{}.eps'.format(t))
