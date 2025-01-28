# Synaptome architecture shapes regional dynamics in the mouse brain
This repository contains code and data in support of "Synaptome architecture shapes regional dynamics in the mouse brain", available as a preprint on [bioRxiv](https://doi.org/10.1101/2025.01.24.634803).
Most of the code was written in Python 3.8.10, with some of the gene ontology analyses being done in Matlab R2022a.
Below, I describe all the folders and files in detail.

## `code`
The [code](code/) folder contains all the code used to run the analyses and generate the figures.
A description of each file follows (in an order that complements the manuscript):
- [scpt_remap_synaptome.py](code/scpt_remap_synaptome.py) will put the synaptome data into the various parcellation schemes I use (e.g. 88 bilateral regions for comparing with fMRI data, 137 right hemisphere regions for comparing with tract-tracing data, ...). This file will also save out the synapse type densities I use in subsequent scripts and analyses.
- [scpt_plot_mouse.py](code/scpt_plot_mouse.py) is a general script handy for plotting mouse brains (and this is the script where I plot much of Figure 1). Note that the package I use to plot mouse brains ([brainglobe-heatmap](https://github.com/brainglobe/brainglobe-heatmap)) requires Python > 3.9, so I had a separate environment where my Python version was 3.9.19 for all the brain plotting. Yes it was annoying, yes there were reasons I stuck to v3.8.10 in my other environment.
- [scpt_hctsa.py](code/scpt_hctsa.py) is the main script for comparing synapse type densities with [hctsa](https://github.com/benfulcher/hctsa) features. Note hctsa is a Matlab toolbox so Matlab was used to compute all the features (this is work that was done by Andrea Luppi). This is also the script where I plot Figure 2.
- [scpt_hctsa_supplement.py](code/scpt_hctsa_supplement.py) is a file where I really dig into the time-series feature list from hctsa for the sake of understanding (and interpreting) the findings. I also do some supplement-type work here, like checking the effects of SNR, motion, and cell type densities. This script produces Figures S2, S3, S7, S8, and S9.
- [scpt_sc.py](code/scpt_sc.py) is where I compare synapse type densities to a weighted and directed [structural connectome](https://www.nature.com/articles/nature13186). This script corresponds to bits of Figure 3.
- [scpt_fc.py](code/scpt_fc.py) is where I compare synapse type densities to a functional connectome, also corresponding to bits of Figure 3.
- [scpt_scfc.py](code/scpt_scfc.py) is where I finally combine SC, FC, and the synaptome, into a structure-function type analysis where I also pull in fMRI data from anaesthetized mice. This script corresponds to Figure 4.
- [scpt_gexp.py](code/scpt_gexp.py) is where I compare synapse type density with gene expression profiles from the Allen Mouse Brain Atlas. It's a supplementary analysis but ended up getting a main text figure for itself (Figure 5). Gene Ontology was done using files from [this Zenodo repository](https://zenodo.org/records/4460714).

## `data`
The [data](data/) folder contains data files used for the analyses.

- All the "mapping" files are used for mapping synaptome regions to SC/FC regions.
- [cellatlas_ero2018.csv](data/cellatlas_ero2018.csv) has cell density data for 9 different cell types (some of the "types" are combinations of other types, e.g. "neurons" which is "inhibitory neurons" and "excitatory neurons" together). This data is from [Ero et al 2018 Front Neuroinformatics](https://doi.org/10.3389/fninf.2018.00084) (Data Sheet 2).
- [synaptome](data/synaptome/) contains synapse type densities in their raw form ([Type_density_Ricky.xlsx](data/mouse_liu2018/Type_density_Ricky.xlsx)) and mapped to the three parcellations I use (synapse density data was shared by Zhen (Ricky) Liu and Seth Grant, see [this](https://doi.org/10.1016/j.neuron.2018.07.007) paper). This folder also contains synapse protein lifetimes, which were derived by [Bulovaite et al](https://doi.org/10.1016/j.neuron.2022.09.009).
- [function](data/function/Gozzi/) contains the BOLD time-series for each mouse and each state ("Awake", and two anaesthetized states, "Halo" and "MedIso"). This also contains the SNR map, and the [hctsa](data/function/Gozzi/HCTSA/) outputs for each mouse.
- [structure](data/structure/) contains the structural connectome file which is originally from the supplement of [Oh et al 2014](https://www.nature.com/articles/nature13186).
- [gene_expression](data/gene_expression/) contains outputs from [abagen](https://abagen.readthedocs.io/en/stable/), including gene expression sampled from coronal slices, sagittal slices, plus some gene information like `entrezID` and `structure_info`.

## `results`
The [results](results/) folder contains some saved outputs from my scripts, including the some spreadsheets containing the hctsa correlations.
The confusing naming convention in [results/HCTSA/](results/HCTSA/) is:

- `hctsa-` because there are correlation results between synapse type densities and hctsa features
- `norm-` because hctsa features were all normalized
- `zscored-` because fMRI time-series were z-scored prior to running hctsa on them
- `noexcl-` because there was an earlier version of this analysis where I was missing one mouse; "no exclusions" means I'm not excluding any of the mice
- `hits-` shows up in the .xlsx spreadsheets and `corrs_` shows up in the .npz files; they mean the same thing
- `p-bonferroni-corrected` is in the spreadsheets because I include a column with Bonferroni-corrected p-values
- `Awake` because the mice are in the awake state

## `manuscript`
The [manuscript](manuscript/) folder contains the PDF of the manuscript as well as Supplementary Table S1 (hctsa correlations).
