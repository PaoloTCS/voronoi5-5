# Knowledge Topology Framework

A Python framework for analyzing semantic relationships between knowledge fragments, identifying knowledge gaps, and designing optimal learning workflows.

## Overview

The Knowledge Topology Framework provides tools for:

1. **Document Processing**: Extract meaningful chunks from documents
2. **Semantic Analysis**: Generate embeddings and analyze relationships between concepts
3. **Knowledge Visualization**: Visualize knowledge as a graph with clusters and gaps
4. **Gap Analysis**: Identify missing concepts and potential areas for content development
5. **Workflow Design**: Create optimized learning paths based on knowledge structure

## Key Concepts

- **Knowledge Graph**: Represents knowledge as nodes (documents, chunks) and edges (relationships)
- **Semantic Gaps**: Areas in the knowledge space that are implied but not explicitly covered
- **Bridge Nodes**: Concepts that connect different knowledge domains
- **Knowledge Flow**: Paths through the knowledge graph optimized for learning

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/knowledge-topology.git
cd knowledge-topology

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
knowledge-topology/
├── models/                  # Core data models
│   ├── chunk.py             # Represents semantic chunks
│   ├── document.py          # Represents documents
│   └── knowledge_graph.py   # Graph representation of knowledge
├── services/                # Business logic services
│   ├── ai/                  # AI services
│   │   ├── embedding_service.py  # Generates text embeddings
│   │   ├── gap_analyzer.py       # Identifies semantic gaps
│   │   ├── text_processor.py     # Processes text into chunks
│   │   └── workflow_designer.py  # Designs learning workflows
│   └── workflow_service.py  # Orchestrates all components
├── visualization/           # Graph visualization tools
│   └── graph_visualizer.py  # Visualizes knowledge graphs
├── ui/                      # User interface
│   ├── app.py               # Streamlit application
│   ├── components/          # UI components
│   └── pages/               # UI pages
├── data/                    # Data storage
├── examples/                # Example usage
│   ├── data/                # Example documents
│   ├── output/              # Example visualizations
│   └── build_ml_knowledge_graph.py  # Example script
└── theory/                  # Theoretical foundation
```

## Usage

### Basic Example

```python
from models.document import Document
from models.knowledge_graph import KnowledgeGraph
from services.ai.text_processor import TextProcessor
from services.ai.embedding_service import EmbeddingService
from visualization.graph_visualizer import GraphVisualizer

# Initialize services
text_processor = TextProcessor()
embedding_service = EmbeddingService()

# Create and process document
document = Document(title="Example Document", content="...")
text_processor.extract_chunks_by_paragraphs(document)

# Create knowledge graph
graph = KnowledgeGraph(name="Example Graph")

# Add document and chunks to graph
# ... (see examples/build_ml_knowledge_graph.py for details)

# Analyze graph
graph.calculate_metrics()
graph.detect_clusters()
gaps = graph.identify_semantic_gaps()

# Visualize graph
visualizer = GraphVisualizer(graph)
visualizer.compute_layout()
viz = visualizer.visualize_matplotlib()
viz.savefig("knowledge_graph.png")
```

### Running the Example

```bash
# Run the machine learning example
python examples/build_ml_knowledge_graph.py
```

### Starting the UI

```bash
# Start the Streamlit UI
streamlit run ui/app.py
```

## Theory

The Knowledge Topology Framework is built on the idea that knowledge can be represented as a topological space, where:

- Documents and chunks form nodes in the space
- Semantic similarity creates connections between nodes
- Clusters represent coherent knowledge domains
- Gaps represent potential missing knowledge
- Learning paths are trajectories through the space

This topological view allows for:
- Identifying structural patterns in knowledge
- Finding optimal learning sequences
- Discovering implicit relationships between concepts
- Pinpointing areas for knowledge development

## Dependencies

- NetworkX (graph algorithms)
- NumPy (numerical operations)
- Matplotlib & Plotly (visualization)
- NLTK (text processing)
- Streamlit (UI)
- Sentence-Transformers or OpenAI API (embeddings)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
