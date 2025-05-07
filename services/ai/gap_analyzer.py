import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from services.ai.embedding_service import EmbeddingService
import networkx as nx
import uuid

class GapAnalyzer:
    """
    Service for analyzing knowledge gaps in the knowledge graph.
    
    This service identifies potential semantic gaps by analyzing:
    1. Triangulation of embedding vectors to find implied but missing concepts
    2. Bridge nodes that connect different clusters
    3. Boundary nodes that mark the edge of knowledge domains
    4. Flow analysis to find bottlenecks in knowledge paths
    """
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize the gap analyzer.
        
        Args:
            embedding_service: Service for generating embeddings
        """
        self.embedding_service = embedding_service or EmbeddingService()
    
    def identify_semantic_gaps(self, knowledge_graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """
        Identify potential semantic gaps in the knowledge graph.
        
        A semantic gap is a region in the embedding space that is implied by the
        surrounding nodes but does not contain any actual nodes. These could represent
        concepts that are logically implied but not explicitly stated.
        
        Args:
            knowledge_graph: The knowledge graph to analyze
            
        Returns:
            List of semantic gap dictionaries
        """
        # Let the knowledge graph handle the gap identification logic
        return knowledge_graph.identify_semantic_gaps()
    
    def analyze_bridge_nodes(self, knowledge_graph: KnowledgeGraph, threshold: float = 0.75) -> List[Dict[str, Any]]:
        """
        Identify bridge nodes that connect different clusters.
        
        Bridge nodes have high betweenness centrality and connect otherwise 
        separate regions of the graph. They are important for knowledge flow.
        
        Args:
            knowledge_graph: The knowledge graph to analyze
            threshold: Percentile threshold for identifying bridge nodes
            
        Returns:
            List of bridge node dictionaries
        """
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(knowledge_graph.graph)
        
        # Sort by betweenness (descending)
        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate threshold value
        threshold_value = np.percentile([v for k, v in sorted_nodes], threshold * 100)
        
        # Filter nodes above threshold
        bridge_nodes = []
        for node_id, centrality in sorted_nodes:
            if centrality >= threshold_value:
                # Get node data
                if node_id in knowledge_graph.nodes:
                    node = knowledge_graph.nodes[node_id]
                    
                    # Get neighbors
                    neighbors = knowledge_graph.get_neighbors(node_id)
                    
                    # Add to bridge nodes
                    bridge_nodes.append({
                        "id": node_id,
                        "node_type": node.node_type,
                        "content_id": node.content_id,
                        "betweenness_centrality": centrality,
                        "neighbors": neighbors
                    })
        
        return bridge_nodes
    
    def analyze_boundary_nodes(self, knowledge_graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """
        Identify boundary nodes that mark the edge of knowledge domains.
        
        Boundary nodes are at the periphery of clusters and have connections
        to nodes in different clusters. They help define the boundary of a
        knowledge domain.
        
        Args:
            knowledge_graph: The knowledge graph to analyze
            
        Returns:
            List of boundary node dictionaries
        """
        # First, make sure clusters are detected
        if not hasattr(knowledge_graph, 'clusters') or not knowledge_graph.clusters:
            knowledge_graph.detect_clusters()
        
        # Create a mapping from node ID to cluster
        node_to_cluster = {}
        for i, cluster in enumerate(knowledge_graph.clusters):
            for node_id in cluster.get('nodes', []):
                node_to_cluster[node_id] = i
        
        # Identify boundary nodes - nodes with edges to nodes in different clusters
        boundary_nodes = []
        for node_id in knowledge_graph.nodes:
            if node_id not in node_to_cluster:
                continue
                
            node_cluster = node_to_cluster[node_id]
            neighbors = knowledge_graph.get_neighbors(node_id)
            
            # Check if any neighbors are in different clusters
            different_cluster_neighbors = []
            for neighbor_id in neighbors:
                if neighbor_id in node_to_cluster and node_to_cluster[neighbor_id] != node_cluster:
                    different_cluster_neighbors.append(neighbor_id)
            
            # If node has neighbors in different clusters, it's a boundary node
            if different_cluster_neighbors:
                node = knowledge_graph.nodes[node_id]
                
                boundary_nodes.append({
                    "id": node_id,
                    "node_type": node.node_type,
                    "content_id": node.content_id,
                    "cluster": node_cluster,
                    "different_cluster_neighbors": different_cluster_neighbors,
                    "boundary_strength": len(different_cluster_neighbors) / len(neighbors) if neighbors else 0
                })
        
        # Sort by boundary strength (descending)
        boundary_nodes.sort(key=lambda x: x["boundary_strength"], reverse=True)
        
        return boundary_nodes
    
    def analyze_knowledge_flow(self, knowledge_graph: KnowledgeGraph, source_nodes: List[str], target_nodes: List[str]) -> Dict[str, Any]:
        """
        Analyze the flow of knowledge from source nodes to target nodes.
        
        This analysis identifies bottlenecks and critical paths in the knowledge flow.
        
        Args:
            knowledge_graph: The knowledge graph to analyze
            source_nodes: List of source node IDs
            target_nodes: List of target node IDs
            
        Returns:
            Dictionary with flow analysis results
        """
        # Create a flow network by copying the knowledge graph
        flow_network = knowledge_graph.graph.copy()
        
        # Set edge capacities based on weights
        for u, v, data in flow_network.edges(data=True):
            flow_network[u][v]['capacity'] = data.get('weight', 1.0)
        
        # Add super source and super target
        flow_network.add_node('super_source')
        flow_network.add_node('super_target')
        
        # Connect super source to source nodes
        for node_id in source_nodes:
            if node_id in flow_network:
                flow_network.add_edge('super_source', node_id, capacity=float('inf'))
        
        # Connect target nodes to super target
        for node_id in target_nodes:
            if node_id in flow_network:
                flow_network.add_edge(node_id, 'super_target', capacity=float('inf'))
        
        # Calculate maximum flow
        try:
            flow_value, flow_dict = nx.maximum_flow(flow_network, 'super_source', 'super_target')
        except nx.NetworkXError:
            # Graph may not be connected
            return {
                "flow_value": 0,
                "bottlenecks": [],
                "critical_paths": []
            }
        
        # Find bottlenecks (min-cut)
        try:
            cut_value, partition = nx.minimum_cut(flow_network, 'super_source', 'super_target')
            reachable, non_reachable = partition
            
            # The bottleneck edges are those that go from reachable to non_reachable
            bottlenecks = []
            for u in reachable:
                for v in non_reachable:
                    if flow_network.has_edge(u, v):
                        if u != 'super_source' and v != 'super_target':
                            bottlenecks.append((u, v, flow_network[u][v]['capacity']))
        except nx.NetworkXError:
            bottlenecks = []
        
        # Find critical paths
        critical_paths = []
        for source in source_nodes:
            for target in target_nodes:
                try:
                    path = nx.shortest_path(flow_network, source, target, weight='capacity')
                    if len(path) > 2:  # Path has intermediate nodes
                        critical_paths.append({
                            "source": source,
                            "target": target,
                            "path": path,
                            "length": len(path) - 1
                        })
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        # Sort critical paths by length (ascending)
        critical_paths.sort(key=lambda x: x["length"])
        
        return {
            "flow_value": flow_value,
            "bottlenecks": bottlenecks,
            "critical_paths": critical_paths
        }
    
    def recommend_new_content(self, knowledge_graph: KnowledgeGraph, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend new content to fill knowledge gaps.
        
        Args:
            knowledge_graph: The knowledge graph to analyze
            top_k: Number of top recommendations to return
            
        Returns:
            List of content recommendations
        """
        recommendations = []
        
        # Get semantic gaps
        semantic_gaps = self.identify_semantic_gaps(knowledge_graph)
        
        for gap in semantic_gaps[:top_k]:
            # Extract vertices (nodes that form the gap)
            vertices = gap.get('vertices', [])
            vertex_nodes = [knowledge_graph.nodes[v] for v in vertices if v in knowledge_graph.nodes]
            
            # Get content related to vertices
            vertex_contents = []
            for node in vertex_nodes:
                if node.node_type == 'document':
                    # If the node is a document, try to get its title
                    vertex_contents.append(f"Document: {node.properties.get('title', node.content_id)}")
                elif node.node_type == 'chunk':
                    # If the node is a chunk, try to get its key point
                    vertex_contents.append(f"Chunk: {node.properties.get('key_point', 'Content chunk')}")
                else:
                    vertex_contents.append(f"{node.node_type}: {node.content_id}")
            
            # Create a recommendation
            recommendation = {
                "id": f"rec_{len(recommendations)}",
                "gap_id": gap.get('id', ''),
                "gap_distance": gap.get('distance', 0),
                "vertices": vertices,
                "vertex_contents": vertex_contents,
                "recommendation": f"Create content that connects the concepts in: {', '.join(vertex_contents)}",
                "confidence": 1.0 - (gap.get('distance', 0) / 2)  # Convert distance to confidence
            }
            
            recommendations.append(recommendation)
        
        # Add recommendations based on bridge nodes
        bridge_nodes = self.analyze_bridge_nodes(knowledge_graph)
        
        for i, bridge in enumerate(bridge_nodes[:min(top_k // 2, len(bridge_nodes))]):
            # Get node data
            node_id = bridge["id"]
            node = knowledge_graph.nodes[node_id]
            
            # Get neighbors
            neighbors = bridge["neighbors"]
            neighbor_nodes = [knowledge_graph.nodes[n] for n in neighbors if n in knowledge_graph.nodes]
            
            # Get content related to neighbors
            neighbor_contents = []
            for n_node in neighbor_nodes[:3]:  # Limit to 3 neighbors
                if n_node.node_type == 'document':
                    neighbor_contents.append(f"Document: {n_node.properties.get('title', n_node.content_id)}")
                elif n_node.node_type == 'chunk':
                    neighbor_contents.append(f"Chunk: {n_node.properties.get('key_point', 'Content chunk')}")
                else:
                    neighbor_contents.append(f"{n_node.node_type}: {n_node.content_id}")
            
            # Create a recommendation based on bridge node
            recommendation = {
                "id": f"rec_{len(recommendations)}",
                "bridge_id": node_id,
                "betweenness_centrality": bridge["betweenness_centrality"],
                "neighbors": neighbors,
                "neighbor_contents": neighbor_contents,
                "recommendation": f"Expand content related to bridge concept that connects: {', '.join(neighbor_contents)}",
                "confidence": bridge["betweenness_centrality"] * 0.8  # Scale to confidence
            }
            
            recommendations.append(recommendation)
        
        # Sort by confidence (descending)
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Return top-k
        return recommendations[:top_k]
    
    def generate_knowledge_path(self, knowledge_graph: KnowledgeGraph, source_concept: str, target_concept: str) -> Dict[str, Any]:
        """
        Generate a structured knowledge path from source concept to target concept.
        
        This is useful for creating learning paths or reasoning chains.
        
        Args:
            knowledge_graph: The knowledge graph to analyze
            source_concept: Text description of source concept
            target_concept: Text description of target concept
            
        Returns:
            Dictionary with knowledge path details
        """
        # Find nodes most similar to source and target concepts
        source_nodes = self._find_most_similar_nodes(knowledge_graph, source_concept)
        target_nodes = self._find_most_similar_nodes(knowledge_graph, target_concept)
        
        if not source_nodes or not target_nodes:
            return {
                "source": source_concept,
                "target": target_concept,
                "path_exists": False,
                "path": [],
                "explanation": "Could not find nodes matching the source or target concepts."
            }
        
        # Get the best source and target nodes
        best_source = source_nodes[0]["id"]
        best_target = target_nodes[0]["id"]
        
        # Find shortest path
        try:
            path = nx.shortest_path(knowledge_graph.graph, best_source, best_target)
            
            # Get path details
            path_details = []
            for i, node_id in enumerate(path):
                if node_id in knowledge_graph.nodes:
                    node = knowledge_graph.nodes[node_id]
                    
                    # Get node details
                    node_detail = {
                        "id": node_id,
                        "node_type": node.node_type,
                        "content_id": node.content_id
                    }
                    
                    # Add relationship to next node
                    if i < len(path) - 1:
                        next_node_id = path[i+1]
                        edge_data = knowledge_graph.graph[node_id][next_node_id]
                        relationship_type = edge_data.get('relationship_type', 'connection')
                        
                        node_detail["next_relationship"] = relationship_type
                    
                    path_details.append(node_detail)
            
            # Generate explanatory steps
            steps = []
            for i, node_detail in enumerate(path_details):
                # Get node content
                node = knowledge_graph.nodes[node_detail["id"]]
                
                # Create step description
                if node.node_type == 'document':
                    step = f"Study document: {node.properties.get('title', node.content_id)}"
                elif node.node_type == 'chunk':
                    step = f"Learn concept: {node.properties.get('key_point', 'Content chunk')}"
                else:
                    step = f"Explore {node.node_type}: {node.content_id}"
                
                # Add relationship to next step
                if i < len(path_details) - 1:
                    relationship = path_details[i]["next_relationship"]
                    step += f" → {relationship} → "
                
                steps.append(step)
            
            return {
                "source": source_concept,
                "target": target_concept,
                "path_exists": True,
                "path": path,
                "path_details": path_details,
                "steps": steps,
                "length": len(path) - 1
            }
            
        except nx.NetworkXNoPath:
            return {
                "source": source_concept,
                "target": target_concept,
                "path_exists": False,
                "path": [],
                "explanation": "No path exists between the source and target concepts."
            }
    
    def _find_most_similar_nodes(self, knowledge_graph: KnowledgeGraph, concept: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find nodes most similar to the given concept.
        
        Args:
            knowledge_graph: The knowledge graph to search
            concept: Concept to find similar nodes for
            top_k: Number of top results to return
            
        Returns:
            List of similar node dictionaries
        """
        # If the embedding service is not available, return an empty list
        if not hasattr(self, 'embedding_service') or not self.embedding_service:
            return []
        
        # Generate embedding for the concept
        concept_embedding = self.embedding_service.get_embedding(concept)
        
        # Compare with node embeddings
        similarities = []
        for node_id, node in knowledge_graph.nodes.items():
            if node.embedding is not None:
                # Calculate cosine similarity
                similarity = np.dot(concept_embedding, node.embedding) / (
                    np.linalg.norm(concept_embedding) * np.linalg.norm(node.embedding)
                )
                
                similarities.append({
                    "id": node_id,
                    "node_type": node.node_type,
                    "content_id": node.content_id,
                    "similarity": float(similarity)
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top-k
        return similarities[:top_k]
