# Control networks during rest

## Introduction

The goal of this notebook is to extract cingulo-opecular (CON) and fronto-parietal (FPN) control networks from julia2018 resting-state dataset. The two networks define control-related connectome with regions from all over the brain. In particular, they forms 4 interesting clusters in CON, FPN, the cerebellum, and TPJ.

Here, I will try to reimplement a method explained in [Dosenbach et al. (2007)]() but using another resting-state dataset. To keep the  results comparable, I will stay faithful to the original paper, meaning the same set of parameters and visualizations will be used.

## Method Summary

I will call the method Dosenbach2007 which is, in simple word, a graph-centric method to identify control networks using resting-state data.

The Dosenbach2007 method first uses a set of pre-defined cognitive control ROIs to extract time-series and then correlate those time-series together to extract seed-based connectivities. Next, it binarizes those connectivities by using an arbitrary threshold `r`. The remaining edges form a small-world network of 8 subgraphs that are related to different aspects of cognitive control.


## Notebooks

TODO

1 Connectome

2 Figure 5

3 Figure 9

4 Hierarchical clustering

6 Graph analysis

99 Chord plot
