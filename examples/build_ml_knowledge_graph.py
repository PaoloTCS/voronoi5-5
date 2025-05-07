#!/usr/bin/env python3
"""
Example script to demonstrate building a knowledge graph from machine learning documents.

This script:
1. Loads sample documents about ML topics
2. Processes them into chunks
3. Creates a knowledge graph
4. Identifies semantic gaps and relationships
5. Generates visualizations
6. Creates a learning workflow
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

# Import our models and services
from models.document import Document
from models.chunk import Chunk
from models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from services.ai.text_processor import TextProcessor
from services.ai.embedding_service import EmbeddingService
from services.ai.gap_analyzer import GapAnalyzer
from services.ai.workflow_designer import WorkflowDesigner, Workflow
from visualization.graph_visualizer import GraphVisualizer

# Directory paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_document(filename):
    """Load a document from the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract title from filename
    title = ' '.join(filename.split('.')[0].split('_')).title()
    
    # Create document
    document = Document(title=title, content=content)
    
    return document

def main():
    print("Knowledge Topology Example: Machine Learning Knowledge Graph")
    print("=" * 80)
    
    # Initialize services
    text_processor = TextProcessor()
    embedding_service = EmbeddingService()  # Using simulated embeddings
    gap_analyzer = GapAnalyzer(embedding_service=embedding_service)
    workflow_designer = WorkflowDesigner(gap_analyzer=gap_analyzer)
    
    # Load documents
    print("\nLoading documents...")
    documents = {}
    for filename in ["intro_to_ml.txt", "neural_networks.txt", "reinforcement_learning.txt"]:
        doc = load_document(filename)
        documents[doc.id] = doc
        print(f"  Loaded: {doc.title} (ID: {doc.id})")
    
    # Process documents to extract chunks
    print("\nExtracting chunks from documents...")
    for doc_id, doc in documents.items():
        # Process document - use different methods for variety
        if "intro_to_ml" in doc.title.lower():
            chunks = text_processor.extract_chunks_by_sections(doc)
        elif "neural" in doc.title.lower():
            chunks = text_processor.extract_chunks_by_paragraphs(doc)
        else:
            chunks = text_processor.extract_chunks_by_fixed_size(doc, max_tokens=150, overlap=30)
        
        # Rank chunks by importance
        text_processor.rank_chunks_by_importance(doc.chunks)
        
        print(f"  Extracted {len(doc.chunks)} chunks from '{doc.title}'")
    
    # Create a knowledge graph
    print("\nCreating knowledge graph...")
    graph = KnowledgeGraph(name="Machine Learning Knowledge Graph")
    
    # Add documents and chunks to the graph
    for doc_id, doc in documents.items():
        # Create document node
        doc_node = KnowledgeNode(
            id=doc.id,
            node_type='document',
            content_id=doc.id,
            properties={
                'title': doc.title,
                'knowledge_domain': 'Machine Learning',
            }
        )
        graph.add_node(doc_node)
        
        # Add chunks for this document
        for chunk in doc.chunks:
            # Generate embedding for chunk
            chunk.embedding = embedding_service.get_embedding(chunk.content)
            
            # Create chunk node
            chunk_node = KnowledgeNode(
                id=chunk.id,
                node_type='chunk',
                content_id=chunk.id,
                embedding=chunk.embedding,
                properties={
                    'key_point': chunk.key_point,
                    'context_label': chunk.context_label,
                    'importance_rank': chunk.importance_rank
                }
            )
            graph.add_node(chunk_node)
            
            # Create edge from document to chunk
            edge_id = f"doc_chunk_{doc.id}_{chunk.id}"
            edge = KnowledgeEdge(
                id=edge_id,
                source_id=doc.id,
                target_id=chunk.id,
                relationship_type='contains',
                weight=1.0
            )
            graph.add_edge(edge)
    
    print(f"  Added {len(graph.nodes)} nodes and {len(graph.edges)} edges to graph")
    
    # Calculate similarity edges between chunks
    print("\nCalculating semantic relationships...")
    similarity_edges_added = 0
    
    # Get chunk nodes
    chunk_nodes = {node_id: node for node_id, node in graph.nodes.items() 
                   if node.node_type == 'chunk' and node.embedding is not None}
    
    # Compare each pair of chunks
    for node1_id, node1 in chunk_nodes.items():
        for node2_id, node2 in chunk_nodes.items():
            if node1_id != node2_id and node1_id in graph.nodes and node2_id in graph.nodes:
                # Skip if already connected
                if graph.graph.has_edge(node1_id, node2_id):
                    continue
                
                # Calculate cosine similarity
                similarity = embedding_service.get_similarity(
                    graph.nodes[node1_id].properties.get('key_point', ''),
                    graph.nodes[node2_id].properties.get('key_point', '')
                )
                
                # Add edge if similarity is above threshold
                if similarity >= 0.6:  # Adjust threshold as needed
                    edge_id = f"sim_{node1_id}_{node2_id}"
                    edge = KnowledgeEdge(
                        id=edge_id,
                        source_id=node1_id,
                        target_id=node2_id,
                        relationship_type='similarity',
                        weight=float(similarity)
                    )
                    graph.add_edge(edge)
                    similarity_edges_added += 1
    
    print(f"  Added {similarity_edges_added} similarity edges")
    
    # Calculate graph metrics
    print("\nCalculating graph metrics...")
    metrics = graph.calculate_metrics()
    print(f"  Graph density: {metrics.get('density', 0):.4f}")
    if metrics.get('is_connected', False):
        print("  Graph is connected")
    else:
        print(f"  Graph has {metrics.get('components', 0)} connected components")
    
    # Detect clusters
    print("\nDetecting knowledge clusters...")
    clusters = graph.detect_clusters()
    print(f"  Detected {len(clusters)} clusters")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {cluster.get('size', 0)} nodes")
    
    # Identify semantic gaps
    print("\nIdentifying semantic gaps...")
    gaps = graph.identify_semantic_gaps()
    print(f"  Detected {len(gaps)} potential semantic gaps")
    
    # Visualize graph
    print("\nCreating visualizations...")
    visualizer = GraphVisualizer(graph)
    
    # Compute layout for visualization
    visualizer.compute_layout(layout_type="force_directed")
    
    # Create main graph visualization
    graph_viz = visualizer.visualize_matplotlib(
        title="Machine Learning Knowledge Graph",
        show_labels=True,
        color_by="node_type"
    )
    graph_viz.savefig(os.path.join(OUTPUT_DIR, "ml_knowledge_graph.png"), dpi=300)
    
    # Create cluster visualization
    cluster_viz = visualizer.visualize_clusters(
        title="Machine Learning Knowledge Clusters"
    )
    cluster_viz.savefig(os.path.join(OUTPUT_DIR, "ml_knowledge_clusters.png"), dpi=300)
    
    # Create gaps visualization
    gaps_viz = visualizer.visualize_semantic_gaps(
        title="Machine Learning Knowledge Gaps"
    )
    if gaps_viz is not None:
        gaps_viz.savefig(os.path.join(OUTPUT_DIR, "ml_knowledge_gaps.png"), dpi=300)
    else:
        print("  No semantic gaps visualization created (no gaps detected)")
    
    # Export graph data for web visualization
    visualizer.export_to_json(os.path.join(OUTPUT_DIR, "ml_knowledge_graph.json"))
    
    # Generate recommendations
    print("\nGenerating gap recommendations...")
    recommendations = gap_analyzer.recommend_new_content(graph)
    for i, rec in enumerate(recommendations[:3]):
        print(f"  Recommendation {i+1}: {rec.get('recommendation', '')}")
        print(f"    Confidence: {rec.get('confidence', 0):.2f}")
        print()
    
    # Create learning workflow
    print("\nDesigning learning workflow...")
    workflow = workflow_designer.create_workflow_from_graph(
        knowledge_graph=graph,
        name="Machine Learning Fundamentals",
        description="A learning path for understanding machine learning concepts",
        max_steps=7
    )
    
    # Optimize workflow
    workflow = workflow_designer.optimize_workflow(
        workflow=workflow,
        optimization_goal="learning"
    )
    
    # Display workflow
    print(f"  Created workflow with {len(workflow.steps)} steps")
    execution_plan = workflow.get_execution_plan()
    print("\nLearning path:")
    for i, step in enumerate(execution_plan):
        print(f"  {i+1}. {step.get('name', '')}")
        print(f"     {step.get('description', '')}")
        if step.get('prerequisites'):
            print(f"     Prerequisites: {', '.join(step.get('prerequisites', []))}")
        print()
    
    # Create learning path for specific topics
    print("\nCreating targeted learning path...")
    learning_path = workflow_designer.create_learning_path(
        knowledge_graph=graph,
        start_concept="Introduction to Machine Learning",
        target_concept="Reinforcement Learning Applications",
        name="ML to RL Learning Path"
    )
    
    # Display learning path
    path_plan = learning_path.get_execution_plan()
    print(f"Learning path from intro to applications ({len(path_plan)} steps):")
    for i, step in enumerate(path_plan):
        print(f"  {i+1}. {step.get('name', '')}")
    
    print("\nComplete! Visualization files saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
