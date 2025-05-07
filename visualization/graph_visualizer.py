import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.cm as cm
from models.knowledge_graph import KnowledgeGraph

class GraphVisualizer:
    """
    Provides various visualization methods for the knowledge graph.
    Supports both matplotlib and Plotly for different visualization needs.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.graph = knowledge_graph.graph
        self.layout = None
        self.node_positions = {}
    
    def compute_layout(self, layout_type: str = "force_directed"):
        """
        Compute the layout for the graph visualization.
        
        Args:
            layout_type: Type of layout to use ('force_directed', 'circular', 'spectral', etc.)
        """
        if layout_type == "force_directed":
            self.layout = nx.spring_layout(self.graph, seed=42)
        elif layout_type == "circular":
            self.layout = nx.circular_layout(self.graph)
        elif layout_type == "spectral":
            self.layout = nx.spectral_layout(self.graph)
        elif layout_type == "kamada_kawai":
            self.layout = nx.kamada_kawai_layout(self.graph)
        else:
            # Default to force directed
            self.layout = nx.spring_layout(self.graph, seed=42)
        
        self.node_positions = self.layout
    
    def visualize_matplotlib(self, 
                           figsize: tuple = (12, 10),
                           title: str = "Knowledge Graph",
                           node_size_factor: float = 100.0,
                           edge_width_factor: float = 1.0,
                           show_labels: bool = True,
                           label_font_size: int = 8,
                           color_by: str = "node_type",
                           save_path: Optional[str] = None):
        """
        Visualize the knowledge graph using Matplotlib.
        
        Args:
            figsize: Figure size in inches
            title: Figure title
            node_size_factor: Factor to multiply node sizes by
            edge_width_factor: Factor to multiply edge widths by
            show_labels: Whether to show node labels
            label_font_size: Font size for node labels
            color_by: Node attribute to use for coloring ('node_type', 'centrality', 'cluster')
            save_path: Path to save the figure to (if None, figure is displayed)
        """
        if not self.layout:
            self.compute_layout()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Extract node attributes for visualization
        node_types = nx.get_node_attributes(self.graph, 'node_type')
        
        # Determine node color mapping
        if color_by == "node_type":
            # Define color map for node types
            type_color_map = {
                'document': 'blue',
                'chunk': 'green',
                'concept': 'red',
                'gap': 'purple'
            }
            node_colors = [type_color_map.get(node_types.get(node, 'unknown'), 'gray') for node in self.graph.nodes()]
            
        elif color_by == "centrality":
            # Use betweenness centrality for coloring
            betweenness = nx.betweenness_centrality(self.graph)
            # Convert to list of values in the order of self.graph.nodes()
            centrality_values = [betweenness.get(node, 0) for node in self.graph.nodes()]
            # Use a colormap
            cmap = cm.get_cmap('viridis')
            node_colors = [cmap(value) for value in centrality_values]
            
        elif color_by == "cluster":
            # Color by community/cluster
            if hasattr(self.knowledge_graph, 'clusters') and self.knowledge_graph.clusters:
                # Create a mapping from node ID to cluster
                node_to_cluster = {}
                for i, cluster in enumerate(self.knowledge_graph.clusters):
                    for node_id in cluster.get('nodes', []):
                        node_to_cluster[node_id] = i
                
                cmap = cm.get_cmap('tab10')
                node_colors = [cmap(node_to_cluster.get(node, 0) % 10) for node in self.graph.nodes()]
            else:
                # Default to type-based coloring if clusters not available
                node_colors = [type_color_map.get(node_types.get(node, 'unknown'), 'gray') for node in self.graph.nodes()]
        else:
            # Default to a single color
            node_colors = 'skyblue'
        
        # Determine node sizes based on degree centrality
        degree_centrality = nx.degree_centrality(self.graph)
        node_sizes = [degree_centrality[node] * node_size_factor * 1000 + 50 for node in self.graph.nodes()]
        
        # Determine edge weights
        edge_weights = [self.graph[u][v].get('weight', 1.0) for u, v in self.graph.edges()]
        edge_widths = [weight * edge_width_factor for weight in edge_weights]
        
        # Draw the graph
        nx.draw_networkx_nodes(
            self.graph, 
            self.layout, 
            node_color=node_colors,
            node_size=node_sizes
        )
        
        nx.draw_networkx_edges(
            self.graph, 
            self.layout,
            width=edge_widths,
            alpha=0.7,
            edge_color='gray'
        )
        
        if show_labels:
            # Create custom labels
            labels = {}
            for node in self.graph.nodes():
                # Determine label based on node type
                node_type = node_types.get(node, 'unknown')
                
                if node_type == 'document':
                    # Get document titles if available
                    if 'content_id' in self.graph.nodes[node]:
                        content_id = self.graph.nodes[node]['content_id']
                        doc = self.knowledge_graph.nodes.get(node)
                        if hasattr(doc, 'title') and doc.title:
                            labels[node] = doc.title
                        else:
                            labels[node] = f"Doc: {content_id[-8:]}"
                    else:
                        labels[node] = f"Doc: {node[-8:]}"
                        
                elif node_type == 'chunk':
                    # Get chunk key point if available
                    if 'content_id' in self.graph.nodes[node]:
                        content_id = self.graph.nodes[node]['content_id']
                        chunk = self.knowledge_graph.nodes.get(node)
                        if hasattr(chunk, 'key_point') and chunk.key_point:
                            labels[node] = chunk.key_point
                        else:
                            labels[node] = f"Chunk: {content_id[-8:]}"
                    else:
                        labels[node] = f"Chunk: {node[-8:]}"
                else:
                    labels[node] = f"{node_type}: {node[-8:]}"
            
            nx.draw_networkx_labels(
                self.graph, 
                self.layout, 
                labels=labels,
                font_size=label_font_size,
                font_color='black',
                font_weight='bold'
            )
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            return plt.gcf()  # Return the figure
    
    def visualize_plotly(self, 
                        title: str = "Interactive Knowledge Graph",
                        node_size_factor: float = 10.0,
                        edge_width_factor: float = 1.0,
                        color_by: str = "node_type",
                        height: int = 800,
                        width: int = 1200):
        """
        Visualize the knowledge graph using Plotly for interactive visualization.
        
        Args:
            title: Figure title
            node_size_factor: Factor to multiply node sizes by
            edge_width_factor: Factor to multiply edge widths by
            color_by: Node attribute to use for coloring ('node_type', 'centrality', 'cluster')
            height: Height of the figure in pixels
            width: Width of the figure in pixels
            
        Returns:
            Plotly figure object
        """
        if not self.layout:
            self.compute_layout()
        
        # Extract node positions
        pos = self.layout
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        # Determine node colors and sizes
        node_types = nx.get_node_attributes(self.graph, 'node_type')
        
        # For node size
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Color mapping
        if color_by == "node_type":
            # Define color map for node types
            type_color_map = {
                'document': 0,
                'chunk': 1,
                'concept': 2,
                'gap': 3,
                'unknown': 4
            }
            color_scale = ['blue', 'green', 'red', 'purple', 'gray']
            
        elif color_by == "centrality":
            betweenness = nx.betweenness_centrality(self.graph)
            
        elif color_by == "cluster":
            # Color by community/cluster
            if hasattr(self.knowledge_graph, 'clusters') and self.knowledge_graph.clusters:
                # Create a mapping from node ID to cluster
                node_to_cluster = {}
                for i, cluster in enumerate(self.knowledge_graph.clusters):
                    for node_id in cluster.get('nodes', []):
                        node_to_cluster[node_id] = i
            
        # Add nodes
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node text/label
            node_type = node_types.get(node, 'unknown')
            if 'content_id' in self.graph.nodes[node]:
                content_id = self.graph.nodes[node]['content_id']
                if node_type == 'document':
                    # Try to get document title
                    doc = self.knowledge_graph.nodes.get(node)
                    if hasattr(doc, 'title') and doc.title:
                        node_text.append(f"Document: {doc.title}")
                    else:
                        node_text.append(f"Document: {content_id[-8:]}")
                elif node_type == 'chunk':
                    # Try to get chunk key point
                    chunk = self.knowledge_graph.nodes.get(node)
                    if hasattr(chunk, 'key_point') and chunk.key_point:
                        node_text.append(f"Chunk: {chunk.key_point}")
                    else:
                        node_text.append(f"Chunk: {content_id[-8:]}")
                else:
                    node_text.append(f"{node_type}: {content_id[-8:]}")
            else:
                node_text.append(f"{node_type}: {node[-8:]}")
            
            # Node size based on degree centrality
            node_size.append(degree_centrality[node] * node_size_factor * 100 + 10)
            
            # Node color
            if color_by == "node_type":
                node_color.append(type_color_map.get(node_type, 4))
            elif color_by == "centrality":
                node_color.append(betweenness.get(node, 0))
            elif color_by == "cluster":
                if hasattr(self.knowledge_graph, 'clusters') and self.knowledge_graph.clusters:
                    node_color.append(node_to_cluster.get(node, 0))
                else:
                    node_color.append(type_color_map.get(node_type, 4))
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis' if color_by != "node_type" else color_scale,
                color=node_color,
                size=node_size,
                line_width=2,
                line=dict(color='black', width=0.5)
            )
        )
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_width = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge width
            weight = self.graph[edge[0]][edge[1]].get('weight', 1.0)
            edge_width.append(weight * edge_width_factor)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=width,
                height=height
            )
        )
        
        return fig
    
    def visualize_clusters(self, 
                         figsize: tuple = (12, 10),
                         title: str = "Knowledge Graph Clusters",
                         node_size_factor: float = 100.0,
                         save_path: Optional[str] = None):
        """
        Visualize the clusters in the knowledge graph.
        
        Args:
            figsize: Figure size in inches
            title: Figure title
            node_size_factor: Factor to multiply node sizes by
            save_path: Path to save the figure to (if None, figure is displayed)
        """
        if not hasattr(self.knowledge_graph, 'clusters') or not self.knowledge_graph.clusters:
            print("No clusters available. Run knowledge_graph.detect_clusters() first.")
            return None
        
        if not self.layout:
            self.compute_layout()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create a mapping from node ID to cluster
        node_to_cluster = {}
        for i, cluster in enumerate(self.knowledge_graph.clusters):
            for node_id in cluster.get('nodes', []):
                node_to_cluster[node_id] = i
        
        # Define colormap
        cmap = cm.get_cmap('tab10')
        
        # Draw nodes
        for i, node in enumerate(self.graph.nodes()):
            x, y = self.layout[node]
            size = nx.degree_centrality(self.graph)[node] * node_size_factor * 1000 + 50
            cluster_id = node_to_cluster.get(node, -1)
            color = cmap(cluster_id % 10) if cluster_id >= 0 else 'gray'
            
            plt.scatter(x, y, s=size, color=color, alpha=0.7, edgecolors='black')
        
        # Draw edges
        for edge in self.graph.edges():
            x0, y0 = self.layout[edge[0]]
            x1, y1 = self.layout[edge[1]]
            
            # Check if edge is within same cluster
            cluster_0 = node_to_cluster.get(edge[0], -1)
            cluster_1 = node_to_cluster.get(edge[1], -1)
            
            if cluster_0 == cluster_1 and cluster_0 >= 0:
                # Intra-cluster edge
                color = cmap(cluster_0 % 10)
                alpha = 0.7
            else:
                # Inter-cluster edge
                color = 'gray'
                alpha = 0.3
            
            plt.plot([x0, x1], [y0, y1], color=color, alpha=alpha)
        
        # Add cluster labels
        for i, cluster in enumerate(self.knowledge_graph.clusters):
            # Find centroid of cluster
            nodes = cluster.get('nodes', [])
            if nodes:
                x_coords = [self.layout[node][0] for node in nodes if node in self.layout]
                y_coords = [self.layout[node][1] for node in nodes if node in self.layout]
                
                if x_coords and y_coords:
                    centroid_x = sum(x_coords) / len(x_coords)
                    centroid_y = sum(y_coords) / len(y_coords)
                    
                    # Add cluster label
                    plt.text(
                        centroid_x, centroid_y,
                        f"Cluster {i+1}\n({len(nodes)} nodes)",
                        fontsize=10,
                        ha='center',
                        va='center',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
                    )
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            return plt.gcf()  # Return the figure
    
    def visualize_semantic_gaps(self,
                              figsize: tuple = (12, 10),
                              title: str = "Semantic Gaps Analysis",
                              save_path: Optional[str] = None):
        """
        Visualize the semantic gaps in the knowledge graph.
        
        Args:
            figsize: Figure size in inches
            title: Figure title
            save_path: Path to save the figure to (if None, figure is displayed)
        """
        if not hasattr(self.knowledge_graph, 'semantic_gaps') or not self.knowledge_graph.semantic_gaps:
            print("No semantic gaps available. Run knowledge_graph.identify_semantic_gaps() first.")
            return None
        
        if not self.layout:
            self.compute_layout()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Draw all nodes and edges first (background)
        # Nodes
        nx.draw_networkx_nodes(
            self.graph, 
            self.layout, 
            node_color='lightgray',
            node_size=50,
            alpha=0.5
        )
        
        # Edges
        nx.draw_networkx_edges(
            self.graph, 
            self.layout,
            width=0.5,
            alpha=0.3,
            edge_color='lightgray'
        )
        
        # Draw semantic gaps
        for i, gap in enumerate(self.knowledge_graph.semantic_gaps):
            vertices = gap.get('vertices', [])
            
            if len(vertices) >= 3:
                # Draw the triangle of vertices
                x_coords = []
                y_coords = []
                
                for vertex in vertices:
                    if vertex in self.layout:
                        x, y = self.layout[vertex]
                        x_coords.append(x)
                        y_coords.append(y)
                
                # Close the polygon
                if x_coords and len(x_coords) >= 3:
                    x_coords.append(x_coords[0])
                    y_coords.append(y_coords[0])
                    
                    # Draw the triangle
                    plt.fill(x_coords, y_coords, alpha=0.2, color='red')
                    
                    # Draw the centroid
                    if 'centroid' in gap:
                        # We can't directly plot the centroid from the embedding space
                        # Instead, calculate the centroid of the vertices in the layout space
                        centroid_x = sum(x_coords[:-1]) / len(x_coords[:-1])
                        centroid_y = sum(y_coords[:-1]) / len(y_coords[:-1])
                        
                        plt.scatter(centroid_x, centroid_y, color='red', s=200, alpha=0.7, 
                                   marker='X', edgecolors='black')
                        
                        plt.text(
                            centroid_x, centroid_y + 0.05,
                            f"Gap {i+1}\nDistance: {gap.get('distance', 0):.3f}",
                            fontsize=8,
                            ha='center',
                            va='center',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
                        )
                
                # Draw the vertices with highlighting
                for vertex in vertices:
                    if vertex in self.layout:
                        x, y = self.layout[vertex]
                        plt.scatter(x, y, color='blue', s=100, alpha=1.0, edgecolors='black')
                
                # Draw edges between vertices
                for i in range(len(vertices)):
                    for j in range(i+1, len(vertices)):
                        if vertices[i] in self.layout and vertices[j] in self.layout:
                            x0, y0 = self.layout[vertices[i]]
                            x1, y1 = self.layout[vertices[j]]
                            plt.plot([x0, x1], [y0, y1], color='blue', alpha=0.7, linestyle='--')
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            return plt.gcf()  # Return the figure
    
    def export_to_json(self, save_path: str):
        """
        Export the graph visualization data to JSON for use in web applications.
        
        Args:
            save_path: Path to save the JSON file to
        """
        import json
        
        if not self.layout:
            self.compute_layout()
        
        # Extract node attributes
        node_types = nx.get_node_attributes(self.graph, 'node_type')
        degrees = dict(self.graph.degree())
        
        # Prepare node data
        nodes = []
        for node in self.graph.nodes():
            x, y = self.layout[node]
            node_type = node_types.get(node, 'unknown')
            content_id = self.graph.nodes[node].get('content_id', '')
            
            # Get node info based on type
            label = node
            title = ""
            if node_type == 'document':
                doc = self.knowledge_graph.nodes.get(node)
                if hasattr(doc, 'title') and doc.title:
                    title = doc.title
                    label = title
            elif node_type == 'chunk':
                chunk = self.knowledge_graph.nodes.get(node)
                if hasattr(chunk, 'key_point') and chunk.key_point:
                    title = chunk.key_point
                    label = title
            
            nodes.append({
                'id': node,
                'x': float(x),
                'y': float(y),
                'type': node_type,
                'content_id': content_id,
                'title': title,
                'label': label,
                'size': degrees[node] + 1
            })
        
        # Prepare edge data
        edges = []
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            weight = data.get('weight', 1.0)
            relationship_type = data.get('relationship_type', 'unknown')
            
            edges.append({
                'source': source,
                'target': target,
                'weight': float(weight),
                'type': relationship_type
            })
        
        # Prepare cluster data
        clusters = []
        if hasattr(self.knowledge_graph, 'clusters') and self.knowledge_graph.clusters:
            for i, cluster in enumerate(self.knowledge_graph.clusters):
                clusters.append({
                    'id': f"cluster_{i}",
                    'size': cluster.get('size', 0),
                    'nodes': cluster.get('nodes', []),
                    'density': cluster.get('density', 0)
                })
        
        # Prepare semantic gaps data
        gaps = []
        if hasattr(self.knowledge_graph, 'semantic_gaps') and self.knowledge_graph.semantic_gaps:
            for i, gap in enumerate(self.knowledge_graph.semantic_gaps):
                gaps.append({
                    'id': gap.get('id', f"gap_{i}"),
                    'vertices': gap.get('vertices', []),
                    'distance': gap.get('distance', 0),
                    'closest_node': gap.get('closest_node', '')
                })
        
        # Combine all data
        graph_data = {
            'nodes': nodes,
            'edges': edges,
            'clusters': clusters,
            'semantic_gaps': gaps,
            'metrics': self.knowledge_graph.metrics if hasattr(self.knowledge_graph, 'metrics') else {}
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        return graph_data
