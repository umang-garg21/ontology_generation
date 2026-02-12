import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

class BeeWingOntology:
    """
    Class for building and analyzing bee wing ontologies from segmented objects.
    """

    def __init__(self, masks: List[np.ndarray] = None, embeddings: np.ndarray = None, image: np.ndarray = None,
                 image_path: str = None, mask_path: str = None, embedding_path: str = None):
        """
        Initialize with either loaded data or paths to files.

        Args:
            masks: List of binary masks (if providing loaded data)
            embeddings: Numpy array of embeddings (if providing loaded data)
            image: Original image array (if providing loaded data)
            image_path: Path to the original wing image (if loading from files)
            mask_path: Path to the segmented masks (numpy array file)
            embedding_path: Path to DINOv2 embeddings (numpy array file)
        """
        if masks is not None and embeddings is not None and image is not None:
            # Use provided loaded data
            self.masks = masks
            self.embeddings = embeddings
            self.image = image
            self.original_image = image.copy()
        elif image_path and mask_path and embedding_path:
            # Load from files
            self.image_path = image_path
            self.mask_path = mask_path
            self.embedding_path = embedding_path

            # Load data
            self.image = cv2.imread(image_path)
            self.original_image = self.image.copy()
            self.masks = np.load(mask_path)
            self.embeddings = np.load(embedding_path)
        else:
            raise ValueError("Either provide loaded data (masks, embeddings, image) or file paths (image_path, mask_path, embedding_path)")

        # Validate data consistency
        assert len(self.masks) == len(self.embeddings), f"Mask count ({len(self.masks)}) and embedding count ({len(self.embeddings)}) mismatch"

        self.graph = None
        self.object_features = []

    def extract_object_features(self, mask: np.ndarray) -> Dict:
        """
        Extract morphological features from a single object mask.

        Args:
            mask: Binary mask of the object

        Returns:
            Dictionary of features
        """
        # Get connected components
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)

        if len(regions) == 0:
            return {}

        # Use the largest region
        region = max(regions, key=lambda r: r.area)

        features = {
            'centroid': region.centroid,
            'area': region.area,
            'perimeter': region.perimeter,
            'bbox': region.bbox,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
            'extent': region.extent,
            'orientation': region.orientation,
            'major_axis_length': region.major_axis_length,
            'minor_axis_length': region.minor_axis_length,
            'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
        }

        return features

    def build_ontology_graph(self, connectivity_threshold: float = 10.0,
                           boundary_overlap_threshold: float = 0.1) -> nx.Graph:
        """
        Build a graph where nodes are segmented objects and edges represent connectivity.

        Args:
            connectivity_threshold: Maximum distance for proximity-based connections
            boundary_overlap_threshold: Minimum boundary overlap for direct connections

        Returns:
            NetworkX graph with object features and connectivity
        """
        self.graph = nx.Graph()
        self.node_features = {}

        # Add nodes with features
        for i, (mask, embedding) in enumerate(zip(self.masks, self.embeddings)):
            features = self.extract_object_features(mask)

            if not features:  # Skip empty masks
                continue

            # Combine morphological features with embedding
            node_data = {
                'features': features,
                'embedding': embedding,
                'mask': mask,
                'node_id': i
            }

            self.graph.add_node(i, **node_data)
            self.node_features[i] = features

        # Add edges based on connectivity
        nodes = list(self.graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i], nodes[j]

                # Calculate different types of connections
                proximity_weight = self._calculate_proximity_connection(
                    self.node_features[node_i], self.node_features[node_j],
                    connectivity_threshold
                )

                boundary_weight = self._calculate_boundary_connection(
                    self.graph.nodes[node_i]['mask'], self.graph.nodes[node_j]['mask'],
                    boundary_overlap_threshold
                )

                # Combine connection types (take maximum weight)
                connection_weight = max(proximity_weight, boundary_weight)

                if connection_weight > 0:
                    self.graph.add_edge(node_i, node_j, weight=connection_weight)

        return self.graph

    def _calculate_proximity_connection(self, features_i: Dict, features_j: Dict,
                                      threshold: float) -> float:
        """
        Calculate connection strength based on centroid proximity.
        """
        centroid_i = np.array(features_i['centroid'])
        centroid_j = np.array(features_j['centroid'])

        distance = np.linalg.norm(centroid_i - centroid_j)

        if distance <= threshold:
            # Weight decreases with distance (closer = stronger connection)
            return max(0, 1.0 - distance / threshold)
        return 0.0

    def _calculate_boundary_connection(self, mask_i: np.ndarray, mask_j: np.ndarray,
                                     overlap_threshold: float) -> float:
        """
        Calculate connection strength based on boundary overlap.
        """
        # Dilate masks slightly to find touching boundaries
        kernel = np.ones((3, 3), np.uint8)
        dilated_i = cv2.dilate(mask_i.astype(np.uint8), kernel, iterations=1)
        dilated_j = cv2.dilate(mask_j.astype(np.uint8), kernel, iterations=1)

        # Find overlapping boundary regions
        overlap = np.logical_and(dilated_i, dilated_j)
        overlap_area = np.sum(overlap)

        # Normalize by the smaller boundary area
        boundary_i = cv2.morphologyEx(mask_i.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        boundary_j = cv2.morphologyEx(mask_j.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)

        min_boundary_area = min(np.sum(boundary_i), np.sum(boundary_j))

        if min_boundary_area > 0:
            overlap_ratio = overlap_area / min_boundary_area
            return overlap_ratio if overlap_ratio >= overlap_threshold else 0.0

        return 0.0

    def visualize_graph(self, figsize: Tuple[int, int] = (12, 8),
                       with_labels: bool = True) -> plt.Figure:
        """
        Visualize the ontology graph with node positions based on centroids.
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_ontology_graph() first.")

        fig, ax = plt.subplots(figsize=figsize)

        # Position nodes based on centroids
        pos = {}
        for node in self.graph.nodes():
            centroid = self.node_features[node]['centroid']
            pos[node] = (centroid[1], -centroid[0])  # Flip y for image coordinates

        # Draw the graph
        nx.draw(self.graph, pos, with_labels=with_labels, ax=ax,
                node_color='lightblue', node_size=300, font_size=8,
                edge_color='gray', width=[self.graph[u][v]['weight'] * 3 for u, v in self.graph.edges()])

        # Add edge labels (weights)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)

        ax.set_title(f"Ontology Graph - {Path(self.image_path).stem}")
        ax.set_aspect('equal')
        plt.tight_layout()

        return fig

    def get_graph_statistics(self) -> Dict:
        """
        Calculate basic graph statistics.
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_ontology_graph() first.")

        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'average_degree': np.mean([d for n, d in self.graph.degree()]),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }

        return stats

def compare_wing_ontologies(wing1: BeeWingOntology, wing2: BeeWingOntology,
                           similarity_threshold: float = 0.8) -> Dict:
    """
    Compare ontologies between two wings using various metrics.

    Args:
        wing1: First wing ontology
        wing2: Second wing ontology
        similarity_threshold: Threshold for considering embeddings similar

    Returns:
        Dictionary with comparison metrics
    """
    # Graph structure comparison
    graph_sim = nx.graph_edit_distance(wing1.graph, wing2.graph) if (
        wing1.graph.number_of_nodes() < 10 and wing2.graph.number_of_nodes() < 10
    ) else None  # Graph edit distance is expensive for large graphs

    # Node feature comparison (embedding similarity)
    embeddings1 = np.array([wing1.graph.nodes[n]['embedding'] for n in wing1.graph.nodes()])
    embeddings2 = np.array([wing2.graph.nodes[n]['embedding'] for n in wing2.graph.nodes()])

    if len(embeddings1) > 0 and len(embeddings2) > 0:
        embedding_sim_matrix = cosine_similarity(embeddings1, embeddings2)
        max_similarities = np.max(embedding_sim_matrix, axis=1)
        avg_embedding_similarity = np.mean(max_similarities)
        embedding_matches = np.sum(max_similarities >= similarity_threshold)
    else:
        avg_embedding_similarity = 0.0
        embedding_matches = 0

    # Structural comparison
    stats1 = wing1.get_graph_statistics()
    stats2 = wing2.get_graph_statistics()

    comparison = {
        'graph_edit_distance': graph_sim,
        'avg_embedding_similarity': avg_embedding_similarity,
        'embedding_matches_above_threshold': embedding_matches,
        'wing1_stats': stats1,
        'wing2_stats': stats2,
        'node_count_difference': abs(stats1['num_nodes'] - stats2['num_nodes']),
        'edge_count_difference': abs(stats1['num_edges'] - stats2['num_edges']),
        'density_difference': abs(stats1['density'] - stats2['density'])
    }

    return comparison

def visualize_comparison(wing1: BeeWingOntology, wing2: BeeWingOntology,
                        figsize: Tuple[int, int] = (20, 8)) -> plt.Figure:
    """
    Create side-by-side visualization of both wing ontologies.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Wing 1
    pos1 = {node: (wing1.node_features[node]['centroid'][1], -wing1.node_features[node]['centroid'][0])
            for node in wing1.graph.nodes()}
    nx.draw(wing1.graph, pos1, with_labels=True, ax=ax1,
            node_color='lightcoral', node_size=300, font_size=8,
            edge_color='gray', width=[wing1.graph[u][v]['weight'] * 3 for u, v in wing1.graph.edges()])
    ax1.set_title(f"Wing 1: {Path(wing1.image_path).stem}")
    ax1.set_aspect('equal')

    # Wing 2
    pos2 = {node: (wing2.node_features[node]['centroid'][1], -wing2.node_features[node]['centroid'][0])
            for node in wing2.graph.nodes()}
    nx.draw(wing2.graph, pos2, with_labels=True, ax=ax2,
            node_color='lightblue', node_size=300, font_size=8,
            edge_color='gray', width=[wing2.graph[u][v]['weight'] * 3 for u, v in wing2.graph.edges()])
    ax2.set_title(f"Wing 2: {Path(wing2.image_path).stem}")
    ax2.set_aspect('equal')

    plt.tight_layout()
    return fig