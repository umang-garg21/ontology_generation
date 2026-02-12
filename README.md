# Ontology & Instance Segmentation

## Overview
A research toolkit for extracting, representing, and comparing the structure of insect (bee) wings. The pipeline converts high-resolution wing images into graph-based ontologies where:

- **Nodes** = segmented wing cells (individual mask regions)
- **Edges** = direct physical adjacency between regions

It integrates the Segment Anything Model (SAM) for automatic instance segmentation and DINOv2 for semantic embeddings. Geometric and topological features are extracted per region, a robust adjacency graph is constructed, and a multi-step decision pipeline determines whether two specimens share a common ontology.

## Key Features
- **Automated Instance Segmentation:** Uses SAM for precise mask generation of wing cells.
- **DINOv2 Embeddings:** Extracts deep visual features (1024-d) for each segmented region.
- **Graph-Based Ontology:** Constructs a graph encoding both geometric and learned features per node.
- **Robust Adjacency Detection:** Uses a three-stage filter (bbox pre-filter, containment rejection, minimum mask gap via distance transform) to connect only truly adjacent regions.
- **Ontology Decision Pipeline:** A four-step statistical pipeline to determine whether two graphs share a common ontology, with no labels required.
- **Visualization:** Segmentation overlays, graph structure, comparison metrics, and decision summaries.

## Pipeline Summary

1. **Segmentation (SAM)** — Automatic mask generation. `process_masks()` filters by area, removes duplicates (IoU threshold), extracts and saves cutouts.

2. **Embedding Extraction (DINOv2)** — Each cutout is resized to 224x224 and passed through DINOv2 ViT-L/14 to produce a 1024-d embedding.

3. **Geometric Feature Extraction** — Per-mask: area, centroid, perimeter, circularity, aspect ratio, solidity, extent, equivalent diameter.

4. **Adjacency Detection** — Three-stage filtering:
   - *Stage 1*: Expanded bounding-box overlap as a fast pre-filter.
   - *Stage 2*: Containment rejection — if one mask is nested inside another (overlap / smaller area > threshold), skip.
   - *Stage 3*: Minimum mask boundary gap via `cv2.distanceTransform`. Only create an edge if the gap between the two masks is within `max_gap` pixels (default 15). Connection strength = 1 - (gap / max_gap).

5. **Graph Construction** — Nodes carry features + embeddings. Edges carry min_gap, connection_strength, and centroid_distance.

6. **Ontology Decision Pipeline** — Given two graphs A and B with no labels:
   - *Step 1 — Coarse filter*: Compare node count ratio, degree distribution L1, density ratio. Reject if wildly different.
   - *Step 2 — Multi-modal matching*: Hungarian matching using four modalities (appearance, shape, topology, spatial). Measure cross-modality agreement.
   - *Step 3 — Permutation test*: Compare real edge agreement to a null distribution of 1000 random permutations. Report p-value and z-score.
   - *Step 4 — Stable core extraction*: Perturb and re-match 500 times. Node pairs that survive in >80% of perturbations form the shared ontology.
   - *Output*: SHARED/DIFFERENT decision, confidence score (average of 1-p_value, modality consensus, and core coverage), and the stable core subgraph.

## Quick Start

1. Create and activate an environment:
   ```bash
   conda create -n big-bee python=3.10 -y
   conda activate big-bee
   pip install -r requirements.txt
   ```

2. Run the full pipeline (requires GPU):
   ```bash
   jupyter lab
   # open and run ontology_generation.ipynb
   ```

3. Output locations:
   - `cutouts/<image_name>/` — cutout PNGs for each segment
   - `output_masks/<image_name>/out.ome_mask_tokens.npy` — DINOv2 embeddings per region

## Tunable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `area` filter | `process_masks()` | 1000 | Minimum mask area in pixels |
| `iou_threshold` | `process_masks()` | 0.85 | Duplicate mask suppression |
| `max_gap` | `compute_adjacency()` | 15 | Max pixel gap for adjacency |
| `containment_threshold` | `compute_adjacency()` | 0.15 | Max overlap ratio before rejecting as nested |
| `n_perms` | `ontology_decision_pipeline()` | 1000 | Permutation test iterations |
| `n_perturbations` | `stable_core()` | 500 | Perturbations for stable core |
| `stability_threshold` | `stable_core()` | 0.80 | Min survival rate for stable pairs |

## Figures

### `figures/ontology_generation.png`
<p align="center"><img src="figures/ontology_generation.png" alt="Wing graph construction" width="800"/></p>

Four-panel view of the constructed wing ontology graphs:
- **Top row**: Spatial overlays — each wing's graph drawn on the actual image, with nodes at segment centroids and edges connecting adjacent regions. Node color encodes circularity (blue = circular, red = elongated) via the RdYlBu colormap. Node size is proportional to segment area. Edge width reflects connection strength.
- **Bottom row**: Spring layouts — the same graphs drawn using a force-directed layout (weighted by connection strength), showing the abstract topology independent of spatial position. Useful for comparing the "shape" of the connectivity pattern between wings.

### `figures/similarity_matrix.png`
<p align="center"><img src="figures/similarity_matrix.png" alt="Similarity matrix" width="600"/></p>

Heatmap of pairwise DINOv2 cosine similarities between all left-wing segments (rows) and all right-wing segments (columns). Bright yellow cells indicate high visual similarity. A clear bright cell per row suggests a strong one-to-one match. Row 16 (bottom) shows uniformly low similarity — this segment has no good visual match in the right wing.

### `figures/ontology_similarity.png`
<p align="center"><img src="figures/ontology_similarity.png" alt="Ontology decision pipeline" width="800"/></p>

Six-panel output of the ontology decision pipeline:
- **Top-left**: Modality agreement heatmap — shows pairwise agreement between the four matching modalities (appearance, shape, topology, spatial). Each cell is the fraction of nodes that two modalities matched to the same partner. Higher values (darker red) indicate stronger cross-modality consensus.
- **Top-center**: Permutation test histogram — the null distribution of edge agreement under 1000 random node assignments (blue bars), with the real edge agreement marked by the red vertical line. Large separation between the red line and the null distribution indicates statistical significance (reported as p-value and z-score in the title).
- **Top-right**: Stable core scatter — each point is a node pair from the stable core, plotted by stability (x-axis: fraction of perturbations where this pair survived) vs cosine similarity (y-axis). Points clustered at stability=1.0 with high similarity are rock-solid correspondences.
- **Bottom-left/center**: Both wing graphs overlaid on their images with the stable core highlighted — green nodes and thick green edges are part of the shared ontology; red nodes and gray edges are not.
- **Bottom-right**: Decision summary — the final SHARED/DIFFERENT verdict with confidence score, p-value, modality consensus, and core coverage.

### `figures/feature_comparison.png`
<p align="center"><img src="figures/feature_comparison.png" alt="Feature comparison" width="800"/></p>

Four-panel comparison of graph-level properties between left and right wings:
- **Top-left**: Degree distribution — fraction of nodes at each degree for both wings (blue = left, orange = right). The L1 distance between the two histograms is reported in the title. Similar distributions indicate similar connectivity patterns.
- **Top-right**: Laplacian spectra — sorted eigenvalues of each graph's Laplacian matrix. Similar curves indicate similar global topology. Spectral similarity is reported in the title.
- **Bottom-left**: Node matching quality — horizontal bars showing the cosine similarity of each matched node pair (from greedy DINOv2 matching), sorted from worst to best. The red dashed line marks the mean. Pairs near the bottom (low similarity) are weak or incorrect matches.
- **Bottom-right**: Shape descriptor comparison — normalized mean values of seven geometric features across both wings. Bars close in height indicate morphological consistency for that descriptor.

## Primary Files
- `ontology_generation.ipynb` — Main pipeline notebook
- `segment_wing.py`, `segment_wing_augmented.py` — Segmentation helper scripts
- `bee_wing_ontology.py` — Ontology utilities

## Citation
If you use this repository in research, please cite:
- Segment Anything Model (SAM) — [paper](https://arxiv.org/abs/2304.02643) & [repo](https://github.com/facebookresearch/segment-anything)
- DINOv2 — [paper](https://arxiv.org/abs/2304.07193) & [repo](https://github.com/facebookresearch/dinov2)
