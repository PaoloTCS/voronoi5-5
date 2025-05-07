import dataclasses
import uuid
from typing import Optional, List, Dict, Any, Set, Tuple
import numpy as np
import networkx as nx

@dataclasses.dataclass
class KnowledgeNode:
    """
    Represents a node in the knowledge graph, which can be a document or chunk.
    """
    id: str
    node_type: str  # 'document' or 'chunk'
    content_id: str  # ID of the document or chunk
    embedding: Optional[np.ndarray] = None
    properties: Dict[str, Any] = dataclasses.field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"KnowledgeNode(id={self.id}, type={self.node_type}, content_id={self.content_id})"


@dataclasses.dataclass
class KnowledgeEdge:
    """
    Represents a relationship (edge) between two nodes in the knowledge graph.
    """
    id: str
    source_id: str  # ID of the source node
    target_id: str  # ID of the target node
    relationship_type: str  # e.g., 'similarity', 'derivation', 'reference'
    weight: float = 1.0  # Edge weight, typically similarity score
    properties: Dict[str, Any] = dataclasses.field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"KnowledgeEdge(source={self.source_id}, target={self.target_id}, type={self.relationship_type}, weight={self.weight:.4f})"


class KnowledgeGraph:
    """
    Represents the topological structure of knowledge as a graph.
    
    This is the central data structure for analyzing the semantic space,
    identifying clusters, boundaries, connections, and gaps in the knowledge.
    """
    
    def __init__(self, name: str = "Knowledge Graph"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.graph = nx.Graph()  # NetworkX graph for analysis
        self.metrics: Dict[str, Any] = {}
        self.semantic_gaps: List[Dict[str, Any]] = []
        self.clusters: List[Dict[str, Any]] = []
        
    def add_node(self, node: KnowledgeNode) -> str:
        """Add a node to the graph and return its ID."""
        self.nodes[node.id] = node
        
        # Add to NetworkX graph
        self.graph.add_node(
            node.id,
            node_type=node.node_type,
            content_id=node.content_id,
            **node.properties
        )
        
        return node.id
    
    def add_edge(self, edge: KnowledgeEdge) -> str:
        """Add an edge to the graph and return its ID."""
        self.edges[edge.id] = edge
        
        # Add to NetworkX graph
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            id=edge.id,
            relationship_type=edge.relationship_type,
            weight=edge.weight,
            **edge.properties
        )
        
        return edge.id
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its connected edges from the graph."""
        if node_id in self.nodes:
            # Remove all connected edges first
            edges_to_remove = [
                edge_id for edge_id, edge in self.edges.items()
                if edge.source_id == node_id or edge.target_id == node_id
            ]
            
            for edge_id in edges_to_remove:
                del self.edges[edge_id]
            
            # Remove from NetworkX graph
            self.graph.remove_node(node_id)
            
            # Remove the node
            del self.nodes[node_id]
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get IDs of all nodes directly connected to the given node."""
        if node_id not in self.nodes:
            return []
        
        return list(self.graph.neighbors(node_id))
    
    def get_connected_subgraph(self, node_ids: List[str]) -> 'KnowledgeGraph':
        """Create a new graph containing only the specified nodes and their connections."""
        subgraph = KnowledgeGraph(name=f"{self.name} Subgraph")
        
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                subgraph.add_node(node)
        
        for edge_id, edge in self.edges.items():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                subgraph.add_edge(edge)
        
        return subgraph
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate various graph metrics for knowledge analysis."""
        metrics = {}
        
        # Basic graph metrics
        metrics['node_count'] = len(self.nodes)
        metrics['edge_count'] = len(self.edges)
        
        if len(self.nodes) > 0:
            # Density - ratio of actual edges to potential edges
            metrics['density'] = nx.density(self.graph)
            
            # Check if graph is connected
            if nx.is_connected(self.graph):
                metrics['is_connected'] = True
                metrics['diameter'] = nx.diameter(self.graph)
            else:
                metrics['is_connected'] = False
                metrics['components'] = nx.number_connected_components(self.graph)
            
            # Centrality measures
            try:
                degree_centrality = nx.degree_centrality(self.graph)
                betweenness_centrality = nx.betweenness_centrality(self.graph)
                closeness_centrality = nx.closeness_centrality(self.graph)
                
                metrics['centrality'] = {
                    'degree': degree_centrality,
                    'betweenness': betweenness_centrality,
                    'closeness': closeness_centrality
                }
                
                # Update node properties with centrality
                for node_id, centrality in degree_centrality.items():
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        node.properties['degree_centrality'] = centrality
                        node.properties['betweenness_centrality'] = betweenness_centrality[node_id]
                        node.properties['closeness_centrality'] = closeness_centrality[node_id]
                
                # Find bridge nodes (high betweenness)
                bridge_threshold = np.percentile(list(betweenness_centrality.values()), 75)
                metrics['bridge_nodes'] = [
                    node_id for node_id, score in betweenness_centrality.items()
                    if score > bridge_threshold
                ]
                
                # Find central nodes (high closeness)
                central_threshold = np.percentile(list(closeness_centrality.values()), 75)
                metrics['central_nodes'] = [
                    node_id for node_id, score in closeness_centrality.items()
                    if score > central_threshold
                ]
                
            except Exception as e:
                metrics['centrality_error'] = str(e)
        
        # Store metrics
        self.metrics = metrics
        
        return metrics
    
    def detect_clusters(self, resolution: float = 1.0) -> List[Dict[str, Any]]:
        """
        Detect communities/clusters in the knowledge graph.
        
        Args:
            resolution: Resolution parameter for community detection,
                        higher values give smaller communities.
        
        Returns:
            List of cluster information dictionaries.
        """
        if len(self.nodes) < 3:
            self.clusters = []
            return []
        
        try:
            # Use Louvain method for community detection
            communities = nx.community.louvain_communities(
                self.graph, 
                weight='weight',
                resolution=resolution
            )
            
            clusters = []
            for i, community in enumerate(communities):
                cluster = {
                    'id': f"cluster_{i}",
                    'size': len(community),
                    'nodes': list(community),
                    'density': nx.density(self.graph.subgraph(community))
                }
                
                # Calculate centroid if embeddings exist
                community_nodes = [self.nodes[node_id] for node_id in community if node_id in self.nodes]
                embeddings = [node.embedding for node in community_nodes if node.embedding is not None]
                
                if embeddings:
                    centroid = np.mean(embeddings, axis=0)
                    cluster['centroid'] = centroid.tolist()
                
                clusters.append(cluster)
            
            # Sort clusters by size (descending)
            clusters.sort(key=lambda x: x['size'], reverse=True)
            
            # Store clusters
            self.clusters = clusters
            
            return clusters
        
        except Exception as e:
            print(f"Error detecting clusters: {e}")
            self.clusters = []
            return []
    
    def identify_semantic_gaps(self) -> List[Dict[str, Any]]:
        """
        Identify potential semantic gaps in the knowledge space.
        
        A semantic gap is a region in the embedding space that is implied by the
        surrounding nodes but does not contain any actual nodes. These could represent
        concepts that are logically implied but not explicitly stated.
        
        Returns:
            List of semantic gap dictionaries.
        """
        if len(self.nodes) < 3:
            self.semantic_gaps = []
            return []
        
        # Get embeddings from nodes
        embeddings = {}
        for node_id, node in self.nodes.items():
            if node.embedding is not None:
                embeddings[node_id] = node.embedding
        
        if len(embeddings) < 3:
            self.semantic_gaps = []
            return []
        
        # Generate all possible triplets of nodes
        gaps = []
        
        node_ids = list(embeddings.keys())
        for i in range(len(node_ids)-2):
            for j in range(i+1, len(node_ids)-1):
                for k in range(j+1, len(node_ids)):
                    # Get embeddings for this triplet
                    node_id1, node_id2, node_id3 = node_ids[i], node_ids[j], node_ids[k]
                    
                    # Calculate centroid
                    triplet_embeddings = np.array([
                        embeddings[node_id1],
                        embeddings[node_id2],
                        embeddings[node_id3]
                    ])
                    
                    centroid = np.mean(triplet_embeddings, axis=0)
                    
                    # Find the closest node to the centroid
                    distances = {}
                    for node_id, embedding in embeddings.items():
                        # Calculate cosine similarity
                        similarity = np.dot(centroid, embedding) / (
                            np.linalg.norm(centroid) * np.linalg.norm(embedding)
                        )
                        distances[node_id] = 1.0 - similarity  # Convert to distance
                    
                    # Sort by distance (ascending)
                    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
                    
                    # Calculate distance metrics
                    centroid_to_closest = sorted_distances[0][1]
                    closest_node_id = sorted_distances[0][0]
                    
                    # If the closest node is far from the centroid and it's not one of the vertices,
                    # we have identified a potential semantic gap
                    if (centroid_to_closest > 0.2 and 
                        closest_node_id not in [node_id1, node_id2, node_id3]):
                        
                        gap = {
                            'id': f"gap_{len(gaps)}",
                            'vertices': [node_id1, node_id2, node_id3],
                            'centroid': centroid.tolist(),
                            'distance': float(centroid_to_closest),
                            'closest_node': closest_node_id
                        }
                        
                        gaps.append(gap)
        
        # Sort gaps by distance (largest first)
        gaps.sort(key=lambda x: x['distance'], reverse=True)
        
        # Keep only the most significant gaps
        significant_gaps = gaps[:min(10, len(gaps))]
        
        # Store gaps
        self.semantic_gaps = significant_gaps
        
        return significant_gaps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the graph to a serializable dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'nodes': {node_id: {
                'id': node.id,
                'type': node.node_type,
                'content_id': node.content_id,
                'properties': node.properties
            } for node_id, node in self.nodes.items()},
            'edges': {edge_id: {
                'id': edge.id,
                'source': edge.source_id,
                'target': edge.target_id,
                'type': edge.relationship_type,
                'weight': edge.weight,
                'properties': edge.properties
            } for edge_id, edge in self.edges.items()},
            'metrics': self.metrics,
            'clusters': self.clusters,
            'semantic_gaps': self.semantic_gaps
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeGraph':
        """Create a graph from a dictionary representation."""
        graph = cls(name=data.get('name', 'Knowledge Graph'))
        graph.id = data.get('id', str(uuid.uuid4()))
        
        # Add nodes
        for node_data in data.get('nodes', {}).values():
            node = KnowledgeNode(
                id=node_data['id'],
                node_type=node_data['type'],
                content_id=node_data['content_id'],
                properties=node_data.get('properties', {})
            )
            graph.add_node(node)
        
        # Add edges
        for edge_data in data.get('edges', {}).values():
            edge = KnowledgeEdge(
                id=edge_data['id'],
                source_id=edge_data['source'],
                target_id=edge_data['target'],
                relationship_type=edge_data['type'],
                weight=edge_data['weight'],
                properties=edge_data.get('properties', {})
            )
            graph.add_edge(edge)
        
        # Add metrics and analysis results
        graph.metrics = data.get('metrics', {})
        graph.clusters = data.get('clusters', [])
        graph.semantic_gaps = data.get('semantic_gaps', [])
        
        return graph
