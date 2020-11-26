# Control networks during rest

## Introduction

The goal of this notebook is to identify cognitive control connectome of the [föcker2018] resting-state dataset. In particular, I expect to see 4 relevant clusters in CON (cingulo-opecular network), FPN (fronto-parietal network), the cerebellum, and TPJ.

Here, I will try to reimplement a method proposed in [@dosenbach2007], but using another resting-state dataset. To keep the  results comparable, I will stay faithful to the original paper and use the same parameters and visualizations.


## Method summary

In brief, I use graph analysis to identify control networks using resting-state activities in a set of previously defined regions of cognitive control.

A systematic review of the cognitive control literature revealed 39 control-related ROIs [@dosenbach2007]. By extracting resting time-series of those ROIs and correlating them, we can produce a seed-based connectivity matrix of size $(39 \times 39)$ for each subject.

Next, I will binarize the connectivity matrices with respect to an arbitrary threshold `r`. The remaining edges, i.e., connectivities stronger than $r$, are expected to form a small-world network of 8 subgraphs that each might relate to a different aspect of cognitive control. See [@dosenbach2007] and notebooks for more detailed explanation of the method.


## Notebooks

This folder contains the following notebooks, each aims a specific analysis. All of notebooks can be previewed in GitLab, but if you want to change and run them by your own, see the [running notebooks](../../docs/running_notebooks.md) guide.

:notebook: :construction: [**1 Connectome**](1_connectome.ipynb): Loads preprocessed rs-fMRI, masks the brain, extracts time-series of the control-related ROIs, and finally calculates seed-based connectivity (i.e., correlations of all time-series pairs). A final connectivity matrix of size $(N_{subjects} \times 39 \times 39)$ will be generated and stored in the  `outputs/` directory.

:notebook: :construction: [**2 Figure 5**](2_figure5.ipynb): Replication of *Figure 5* of [@dosenbach2007] which illustrates distribution of averaged connectivity values of all subjects together with an arbitrary binarization threshold ($r$). The notebook also contains some additional analyses from [@cohen2014] to find a more justifiable value for $r$, including "keeping k strongest edges" and "one standard deviation from the median".

:notebook: :construction: [**3 Figure 9**](3_figure9.ipynb): An additional confirmatory analysis to verify the reliability of the discovered networks using bootstrapped hierarchical clustering.

:notebook: :construction: [**4 Hierarchical clustering**](4_hierarchical_clustering.ipynb): More complicated and somehow exploratory clustering of both binarized and correlational connectivities into sub-graphs.

:notebook: :construction: [**6 Graph analysis**](6_graph_analysis.ipynb): This notebook explores the characteristics of the discovered networks such as small-world-ness. These analyses were not part of the original method, but are relevant hence enrich the findings.

:notebook: :construction: [**99 Chord plot**](99_chord_plot.ipynb): Visualization experiments to show connectivities and networks using Chord plots.


## References

[@dosenbach2007] https://doi.org/10.1073/pnas.0704320104