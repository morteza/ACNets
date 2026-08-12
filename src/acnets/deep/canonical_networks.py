"""A declared crosswalk from each atlas's networks to one canonical vocabulary.

`acnets.multiverse` can only measure attribution agreement within a single atlas,
because agreement needs a shared feature vocabulary and no mapping exists between
one parcellation's networks and another's. That restriction is recorded in
`scripts/run_agreement.py` and it is the reason the paper's agreement analysis
runs on Dosenbach 2010 alone -- which the variance decomposition then has to
caveat, since a grid containing one atlas cannot say the atlas does not matter.

This module supplies the missing mapping, as data rather than as an algorithm.
Nine of the cached parcellations carry network labels; each label is assigned by
hand to one of seven canonical networks, or to nothing. Hand-assignment is the
honest method here: the alternative -- matching networks by spatial overlap or by
correlating their timeseries -- would derive the crosswalk from the same 32
participants the model is then fitted to, which is the circularity this project
spends most of its length avoiding.

The seven canonical networks are the ones that recur across taxonomies:

    VIS  visual          SMN  somatomotor       CON  cingulo-opercular/salience
    FPN  fronto-parietal DMN  default mode      DAN  dorsal attention
    CER  cerebellum

**Three judgement calls, declared rather than buried.**

Ventral attention is merged into CON. Yeo-derived atlases label a single
`SalVentAttn` network, Gordon and Seitzman separate `Salience`, `VentralAttn` and
`CinguloOperc`, and Dosenbach's CON is closest to the union. Merging is the
correspondence the field usually assumes; it is not the only defensible one, and
`STRICT` below drops the ambiguous members instead so the choice can be varied.

Auditory is dropped rather than folded into SMN. It is a separate network in
Gordon, Seitzman and MSDL, has no counterpart in Dosenbach 2010 or the Yeo
seven-network solution, and folding it into somatomotor would change what SMN
means in exactly the atlas comparison this crosswalk exists to enable.

Labels with no clear counterpart are dropped, not guessed: Gordon's
`CinguloParietal` and `RetrosplenialTemporal`, Seitzman's `MedialTemporalLobe`,
`ParietoMedial` and `Reward`, Schaefer's `Limbic`, DiFuMo's `TempPar`, MSDL's
`Basal`, `Language` and `Temporal`, and Dosenbach 2007's numbered networks. Every
atlas therefore reports coverage -- what fraction of its regions survive the
crosswalk -- and a model using it should report that number alongside its
accuracy.

Cerebellum exists in only three of the nine atlases, so CER edges are absent from
most inputs. This is handled as missingness rather than as zero.
"""

from __future__ import annotations

import numpy as np

CANONICAL = ('VIS', 'SMN', 'CON', 'FPN', 'DMN', 'DAN', 'CER')

# Ambiguous under the ventral-attention judgement call; dropped when strict.
AMBIGUOUS = frozenset({
    'SalVentAttn', 'SalVentAttnA', 'SalVentAttnB',   # Yeo/Schaefer/DiFuMo merge
    'VentralAttn', 'VentralAttention',               # Gordon / Seitzman
    'Salience',                                      # Gordon / Seitzman / MSDL
    'L V Att', 'R V Att',                            # MSDL
})

CROSSWALK: dict[str, dict[str, str]] = {
    'dosenbach2010': {
        'CER': 'CER', 'CON': 'CON', 'DMN': 'DMN',
        'FPN': 'FPN', 'SMN': 'SMN', 'VIS': 'VIS',
    },
    'dosenbach2007': {
        'cerebellum': 'CER',
        'cingulo-opercular': 'CON',
        'fronto-parietal': 'FPN',
        # network-5/6/7, network-tpj and network-vmpfc are unlabelled or
        # region-named rather than network-named; not guessed.
    },
    'gordon2014_2mm': {
        'Visual': 'VIS',
        'SMhand': 'SMN', 'SMmouth': 'SMN',
        'CinguloOperc': 'CON', 'Salience': 'CON', 'VentralAttn': 'CON',
        'FrontoParietal': 'FPN',
        'Default': 'DMN',
        'DorsalAttn': 'DAN',
    },
    'seitzman2018': {
        'Visual': 'VIS',
        'SomatomotorDorsal': 'SMN', 'SomatomotorLateral': 'SMN',
        'CinguloOpercular': 'CON', 'Salience': 'CON', 'VentralAttention': 'CON',
        'FrontoParietal': 'FPN',
        'DefaultMode': 'DMN',
        'DorsalAttention': 'DAN',
    },
    'schaefer100-7': {
        'Vis': 'VIS', 'SomMot': 'SMN', 'SalVentAttn': 'CON',
        'Cont': 'FPN', 'Default': 'DMN', 'DorsAttn': 'DAN',
    },
    'schaefer200-7': {
        'Vis': 'VIS', 'SomMot': 'SMN', 'SalVentAttn': 'CON',
        'Cont': 'FPN', 'Default': 'DMN', 'DorsAttn': 'DAN',
    },
    'difumo_64_2mm': {
        'VisCent': 'VIS', 'VisPeri': 'VIS',
        'SomMotA': 'SMN', 'SomMotB': 'SMN',
        'SalVentAttnA': 'CON', 'SalVentAttnB': 'CON',
        'ContA': 'FPN', 'ContB': 'FPN', 'ContC': 'FPN',
        'DefaultA': 'DMN', 'DefaultB': 'DMN', 'DefaultC': 'DMN',
        'DorsAttnA': 'DAN', 'DorsAttnB': 'DAN',
    },
    'difumo_128_2mm': {
        'VisCent': 'VIS', 'VisPeri': 'VIS',
        'SomMotA': 'SMN', 'SomMotB': 'SMN',
        'SalVentAttnA': 'CON', 'SalVentAttnB': 'CON',
        'ContA': 'FPN', 'ContB': 'FPN', 'ContC': 'FPN',
        'DefaultA': 'DMN', 'DefaultB': 'DMN', 'DefaultC': 'DMN',
        'DorsAttnA': 'DAN', 'DorsAttnB': 'DAN',
    },
    'msdl': {
        'Striate': 'VIS', 'Occ post': 'VIS', 'Vis Sec': 'VIS',
        'Motor': 'SMN',
        'Cing-Ins': 'CON', 'Salience': 'CON', 'L V Att': 'CON', 'R V Att': 'CON',
        'DMN': 'DMN', 'Dors PCC': 'DMN',
        'D Att': 'DAN', 'Ant IPS': 'DAN',
        'Cereb': 'CER',
    },
}


def assignment_matrix(atlas: str,
                      region_networks,
                      strict: bool = False) -> tuple[np.ndarray, dict]:
    """Region-to-canonical-network indicator, and a coverage report.

    Parameters
    ----------
    atlas
        Key into `CROSSWALK`.
    region_networks
        Per-region network label, in the atlas's own vocabulary and in region
        order.
    strict
        Drop the ventral-attention family rather than merging it into CON, so
        the judgement call declared in the module docstring can be varied
        rather than assumed.

    Returns
    -------
    matrix : ndarray (n_regions, 7)
        Binary; a row is all-zero when the region's network has no canonical
        counterpart. Column order is `CANONICAL`.
    report : dict
        `coverage` is the fraction of regions assigned, `present` lists the
        canonical networks this atlas can populate, and `dropped` counts the
        labels that were deliberately not guessed. A model reading this matrix
        should report `coverage`, because an atlas contributing a third of its
        regions is a different measurement from one contributing all of them.
    """
    if atlas not in CROSSWALK:
        raise KeyError(f'no crosswalk for {atlas!r}; '
                       f'known: {sorted(CROSSWALK)}')
    mapping = CROSSWALK[atlas]
    labels = [str(x) for x in region_networks]

    matrix = np.zeros((len(labels), len(CANONICAL)), dtype=float)
    dropped: dict[str, int] = {}
    for row, label in enumerate(labels):
        target = mapping.get(label)
        if strict and label in AMBIGUOUS:
            target = None
        if target is None:
            dropped[label] = dropped.get(label, 0) + 1
            continue
        matrix[row, CANONICAL.index(target)] = 1.0

    present = [name for i, name in enumerate(CANONICAL) if matrix[:, i].any()]
    report = {
        'atlas': atlas,
        'n_regions': len(labels),
        'n_assigned': int(matrix.sum()),
        'coverage': float(matrix.sum() / len(labels)) if labels else 0.0,
        'present': present,
        'missing': [n for n in CANONICAL if n not in present],
        'dropped': dict(sorted(dropped.items(), key=lambda kv: -kv[1])),
        'strict': strict,
    }
    return matrix, report


def edge_names() -> list[str]:
    """Canonical network-pair names, upper triangle including the diagonal."""
    return [f'{CANONICAL[i]} ↔ {CANONICAL[j]}'
            for i in range(len(CANONICAL))
            for j in range(i, len(CANONICAL))]


def pool_connectivity(connectivity: np.ndarray,
                      matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average a region-by-region matrix into canonical network pairs.

    Parameters
    ----------
    connectivity
        (n_subjects, n_regions, n_regions) or (n_regions, n_regions).
    matrix
        Region-to-canonical indicator from `assignment_matrix`.

    Returns
    -------
    pooled : ndarray (n_subjects, 28)
        Mean connectivity over the region pairs spanning each network pair,
        upper triangle including the diagonal. Zero where the pair is
        unpopulated, with `mask` saying which those are.
    mask : ndarray (28,)
        True where the network pair exists in this atlas. The distinction
        matters: an unpopulated pair is missing, not zero-valued, and pooling
        this way -- region connectivity first, then averaging into networks --
        is the "connect then average" order, which section 3.2 of the manuscript
        shows is worth 0.13 to 0.19 less accuracy than the reverse. It is used
        here because the reverse order cannot be expressed as a layer over a
        connectivity matrix, which is the input this model is specified to take.
    """
    single = connectivity.ndim == 2
    if single:
        connectivity = connectivity[None]

    # how many region pairs feed each network pair: |k| x |l|
    sizes = matrix.sum(axis=0)
    counts = np.outer(sizes, sizes)

    sums = np.einsum('rk,srt,tl->skl', matrix, connectivity, matrix)
    with np.errstate(invalid='ignore', divide='ignore'):
        averaged = np.where(counts > 0, sums / np.maximum(counts, 1e-12), 0.0)

    rows, cols = np.triu_indices(len(CANONICAL))
    pooled = averaged[:, rows, cols]
    mask = counts[rows, cols] > 0
    return (pooled[0] if single else pooled), mask
