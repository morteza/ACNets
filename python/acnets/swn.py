"""Small-world network (SWN) analyses functions."""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def graph_smallworld_features(G):
  """Returns a tuple of (C,L) to be used for SWM sigma calculations."""

  largest_subgraph = max(nx.connected_components(G), key=len)
  largest_subgraph = G.subgraph(largest_subgraph)

  clustering = nx.average_clustering(largest_subgraph)
  shortest_path_length = nx.average_shortest_path_length(largest_subgraph)
  return (clustering, shortest_path_length)


def connectome_smallworld_features(connectome, threshold):

  # binarize the connectome
  bin_conn = (np.abs(connectome) >= threshold).astype('int')

  g_control = nx.from_numpy_matrix(bin_conn)                   # small-world (task control)
  g_random = nx.smallworld.random_reference(g_control, niter=10)     # random
  g_lattice = nx.smallworld.lattice_reference(g_control, niter=10)    # lattice

  _features = {
      'control': graph_smallworld_features(g_control),
      'random': graph_smallworld_features(g_random),
      'lattice': graph_smallworld_features(g_lattice),
  }

  return _features


def plot_clustering_coefficient(conn, label, regions, ax=None):

  # collects clustering coefficients for each threshold; format: ('node','threshold_std','r','C')
  C = []

  for x in np.linspace(0.0, 3.0, 50):

    threshold = np.median(conn) + x * np.std(conn)

    # binarize and create graph
    bin_conn = np.abs(conn >= threshold).astype('int')
    Gp = nx.from_numpy_matrix(bin_conn)

    for i, _ in enumerate(regions):
      Ci = nx.average_clustering(Gp, nodes=[i])
      C.append((regions[i], x, threshold, Ci))

  plot_data = pd.DataFrame(C, columns=['node', 'threshold_std', 'threshold', 'C'])

  if ax is None:
    _, ax = plt.subplots(1, 1)

  # average clustering coefficient across all nodes
  C_avg = plot_data.groupby('threshold_std')['C'].mean()

  # plot average clustering coefficients of the nodes
  sns.lineplot(
      data=plot_data,
      x='threshold_std', y='C',
      size='node', alpha=.1, color='gray',
      ax=ax, legend=False
  )

  # aggregated coefficient across all nodes
  sns.lineplot(
      x=plot_data['threshold_std'].unique(),
      y=C_avg,
      alpha=1, color='black',
      ax=ax
  )

  ax.set(
      xlabel='Threshold (number of standard deviations above median)',
      ylabel='Clustering coefficient (C)',
      title=f'[{label.upper()}] Effect of binarization threshold on clustering coefficient\n'
            f'$M={np.median(conn):.2f}, \sigma={np.std(conn):.2f}$'
  )


def permute_swn(G, n_simulations):
  """simulates random network SWM for n times."""
  _sigmas = []
  for _ in tqdm(range(n_simulations), desc='simulating random networks'):
    # step 1:
    Gr = nx.smallworld.random_reference(G, niter=2, connectivity=False)
    # step 2:
    # Gr is already randomized so in calculating Cr and Lr we don't need to do randomize
    sigma = nx.smallworld.sigma(Gr, niter=2, nrand=2)  # must be > 1 for SW
    # omega = nx.smallworld.omega(g) # must be close to 0 for SW

    _sigmas.append(sigma)
  return _sigmas


def swn_permutation_test(conn, n_simulations=2, plot=True, title=None):
  """[summary]

  Args:
      conn ([type]): [description]
      n_simulations (int, optional): [description]. Defaults to 2.
      plot (bool, optional): [description]. Defaults to True.
      title ([type], optional): [description]. Defaults to None.

  Returns:
      (list,float): a tuple of (simulated sigma values, observed sigma)
  """

  threshold = np.median(conn) + np.std(conn)

  # make a binary graph
  bin_conn = (np.abs(conn) >= threshold).astype('int')
  g = nx.from_numpy_matrix(bin_conn)

  # to avoid error when calculating sigma, we need to make sure that the graph is connected
  if not nx.is_connected(g):
    largest_subgraph = max(nx.connected_components(g), key=len)
    g = g.subgraph(largest_subgraph)

  # observed SWN
  print(f'calculating observed SWN in {title}...')
  observed_swn = nx.smallworld.sigma(g, niter=10, nrand=2)  # using NX default parameter values
  print(f'observed SWN = {observed_swn:.2f} (not normalized)')

  # permuted SWN distribution
  print('Now simulating random networks...')
  simulated_swns = permute_swn(g, n_simulations)

  # PLOT
  if plot:
    fig, ax = plt.subplots(figsize=(6, 3))

    simulated_swns_avg = np.mean(simulated_swns)
    simulated_swns_std = np.std(simulated_swns)
    simulated_swn_zscores = (np.array(simulated_swns) - simulated_swns_avg) / simulated_swns_std
    _observed_swn_norm = (observed_swn - simulated_swns_avg) / simulated_swns_std

    sns.kdeplot(simulated_swn_zscores, ax=ax, color='darkgreen')

    ax.vlines(
        _observed_swn_norm,
        ymin=0, ymax=1.0,
        color='black')

    ax.set(xlabel='$\sigma_{\t{swn}}$')

    plt.legend([f'Permuted values (N={n_simulations})', 'Observed value (normalized)'])

    if title is not None:
      plt.suptitle(title)

    plt.show()

  return simulated_swns, observed_swn
