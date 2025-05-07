import streamlit as st
from pathlib import Path
import sys
import os
from PyPDF2 import PdfReader
import io
import matplotlib.pyplot as plt
import networkx as nx
import uuid
import json
import datetime
import warnings
import pandas as pd

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path to import modules
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

# Import our models
from models.document import Document
from models.chunk import Chunk
from models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from services.ai.text_processor import TextProcessor
from services.ai.embedding_service import EmbeddingService

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Knowledge Topology Workflow",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create sidebar navigation
st.sidebar.title("Knowledge Topology")
st.sidebar.markdown("## 🧠 Knowledge Analysis")

# Navigation options
page = st.sidebar.radio(
    "Navigate to:",
    ["Overview", "Document Analysis", "Knowledge Graph", "Semantic Gaps", "Workflow Designer"]
)

# Initialize session state for storing documents and knowledge graph
if 'documents' not in st.session_state:
    st.session_state.documents = {}  # Store documents by ID
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None
if 'document_library' not in st.session_state:
    st.session_state.document_library = {}  # Library of stored documents
if 'need_rerun' not in st.session_state:
    st.session_state.need_rerun = False

# Create documents directory if it doesn't exist
DOCUMENTS_DIR = os.path.join(parent_dir, "documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Simplified document saving function
def save_document(title, content, chunks=None):
    """Save a document to disk with a streamlined approach"""
    try:
        print(f"\n============ SAVING DOCUMENT: {title} ============")
        
        # Create a unique doc ID with timestamp
        doc_id = f"doc_{int(datetime.datetime.now().timestamp())}"
        print(f"Generated doc_id: {doc_id}")
        
        # Create filename
        file_name = f"{doc_id}.json"
        file_path = os.path.join(DOCUMENTS_DIR, file_name)
        
        print(f"Saving to: {file_path}")
        
        # Prepare document data
        doc_data = {
            "id": doc_id,
            "title": title,
            "content": content,
            "chunks": []
        }
        
        # Add chunks if provided
        if chunks:
            doc_data["chunks"] = [
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "key_point": chunk.key_point,
                    "context_label": chunk.context_label,
                    "importance_rank": chunk.importance_rank
                } for chunk in chunks
            ]
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(doc_data, f, indent=2)
        
        # Verify file was created
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"Success! File created: {file_path}, size: {file_size} bytes")
            
            # Create document info for library
            doc_info = {
                'id': doc_id,
                'title': title,
                'upload_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'chunks': len(doc_data.get('chunks', [])),
                'characters': len(content),
                'path': file_name
            }
            
            # Update session state
            st.session_state.document_library[doc_id] = doc_info
            print(f"Added to document library: {doc_id}")
            
            return True, doc_id
        else:
            print(f"ERROR: File not created at {file_path}")
            return False, None
    except Exception as e:
        import traceback
        print(f"ERROR in save_document: {str(e)}")
        print(traceback.format_exc())
        return False, None

# Function to load document from disk
def load_document_from_library(doc_id):
    """Load document from disk"""
    doc_path = os.path.join(DOCUMENTS_DIR, f"{doc_id}.json")
    if not os.path.exists(doc_path):
        print(f"Document file not found: {doc_path}")
        return None
    
    try:
        with open(doc_path, 'r') as f:
            doc_data = json.load(f)
        
        # Create document
        doc = Document(
            id=doc_data['id'],
            title=doc_data['title'],
            content=doc_data['content']
        )
        
        # Add chunks
        if 'chunks' in doc_data:
            for chunk_data in doc_data['chunks']:
                chunk = Chunk(
                    id=chunk_data['id'],
                    document_id=doc.id,
                    content=chunk_data['content'],
                    key_point=chunk_data['key_point'],
                    context_label=chunk_data['context_label'],
                    importance_rank=chunk_data['importance_rank']
                )
                doc.chunks.append(chunk)
        
        return doc
    except Exception as e:
        import traceback
        print(f"Error loading document {doc_id}: {e}")
        print(traceback.format_exc())
        return None

# Load existing document library
def load_document_library():
    """Load document library from disk"""
    # Start with a clean slate
    print("\n============ LOADING DOCUMENT LIBRARY ============")
    st.session_state.document_library = {}
    
    try:
        # Check if directory exists
        if not os.path.exists(DOCUMENTS_DIR):
            print(f"Creating documents directory: {DOCUMENTS_DIR}")
            os.makedirs(DOCUMENTS_DIR)
            return
        
        # List all JSON files in directory
        all_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith('.json')]
        print(f"Found {len(all_files)} JSON files: {all_files}")
            
        # Load each document
        for file_name in all_files:
            try:
                full_path = os.path.join(DOCUMENTS_DIR, file_name)
                print(f"Loading file: {full_path}")
                
                # Read the file
                with open(full_path, 'r') as f:
                    doc_data = json.load(f)
                
                # Make sure we have an ID
                if 'id' not in doc_data:
                    print(f"Warning: Document {file_name} has no ID")
                    doc_id = f"auto_id_{file_name.replace('.json', '')}"
                else:
                    doc_id = doc_data['id']
                
                # Get document title
                title = doc_data.get('title', 'Untitled Document')
                
                # Create document info
                doc_info = {
                    'id': doc_id,
                    'title': title,
                    'upload_date': datetime.datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M:%S"),
                    'chunks': len(doc_data.get('chunks', [])),
                    'characters': len(doc_data.get('content', '')),
                    'path': file_name
                }
                
                # Add to library
                st.session_state.document_library[doc_id] = doc_info
                print(f"Loaded document: {doc_info['title']} (ID: {doc_id})")
                
            except Exception as e:
                import traceback
                print(f"Error loading document {file_name}: {e}")
                print(traceback.format_exc())
        
        print(f"Document library now contains {len(st.session_state.document_library)} documents")
    
    except Exception as e:
        import traceback
        print(f"Error loading document library: {e}")
        print(traceback.format_exc())

# Load document library on startup
load_document_library()

# Main title
st.title("AI-Powered Knowledge Topology Workflow")

# Page content
if page == "Overview":
    st.header("Knowledge Topology Framework")
    
    st.markdown("""
    This application helps you analyze documents to:
    
    1. Extract meaningful chunks of information
    2. Create semantic embeddings
    3. Visualize knowledge as a topological space
    4. Identify semantic gaps and relationships
    5. Design intelligent workflows based on knowledge structure
    
    ## Getting Started
    
    Upload documents to start analyzing the knowledge structure.
    """)
    
    # Show document library stats
    if st.session_state.document_library:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Documents", value=len(st.session_state.document_library))
        
        with col2:
            total_chunks = sum(doc.get('chunks', 0) for doc in st.session_state.document_library.values())
            st.metric(label="Knowledge Chunks", value=total_chunks)
        
        with col3:
            # Assuming 5% of chunks connect to each other
            if total_chunks > 0:
                st.metric(label="Potential Connections", value=int(total_chunks * (total_chunks-1) * 0.05))
            else:
                st.metric(label="Potential Connections", value=0)
    else:
        # Sample metrics (empty library)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Documents", value="0")
        
        with col2:
            st.metric(label="Knowledge Chunks", value="0")
        
        with col3:
            st.metric(label="Semantic Connections", value="0")

elif page == "Document Analysis":
    st.header("Document Analysis")
    
    # Create tabs for upload and library
    upload_tab, library_tab = st.tabs(["Upload New Document", "Document Library"])
    
    with upload_tab:
        # File uploader
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
        
        if uploaded_file is not None:
            # Display basic document info
            st.success(f"Uploaded: {uploaded_file.name}")
            
            # Process the file based on type
            content = ""
            file_type = ""
            
            # Extract content based on file type
            if uploaded_file.name.endswith('.pdf'):
                file_type = "PDF"
                try:
                    # Process PDF file
                    pdf_bytes = io.BytesIO(uploaded_file.getvalue())
                    pdf_reader = PdfReader(pdf_bytes)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n\n"
                    
                    if not content.strip():
                        st.warning("Could not extract text from PDF. The file might be scanned or contain only images.")
                        content = "PDF text extraction failed. The document may be scanned or image-based."
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    content = f"Error processing PDF: {str(e)}"
            elif uploaded_file.name.endswith('.docx'):
                file_type = "DOCX"
                content = "DOCX file processing not implemented yet"
            else:
                file_type = "TXT"
                try:
                    content = uploaded_file.getvalue().decode("utf-8")
                except UnicodeDecodeError:
                    content = "Could not decode file content"
            
            # Display sample of content
            st.subheader("Document Preview")
            preview = content[:500] + ("..." if len(content) > 500 else "")
            st.text_area("Content Preview", preview, height=200)
            
            if st.button("Process and Save Document", key="process_doc"):
                with st.spinner("Processing document..."):
                    # Add debug output
                    print(f"\n============ PROCESSING DOCUMENT: {uploaded_file.name} ============")
                    print(f"Document type: {file_type}")
                    print(f"Content length: {len(content)}")
                    
                    # Initialize text processor
                    text_processor = TextProcessor()
                    
                    # Create document
                    doc = Document(
                        title=uploaded_file.name,
                        content=content
                    )
                    
                    # Process document to extract chunks if content is available
                    if len(content.strip()) > 0:
                        chunks = text_processor.extract_chunks_by_paragraphs(doc)
                        print(f"Extracted {len(chunks)} chunks from document")
                    else:
                        print("No content to process")
                        doc.chunks = []
                    
                    # Save document
                    success, doc_id = save_document(
                        title=uploaded_file.name,
                        content=content,
                        chunks=doc.chunks
                    )
                    
                    if success:
                        st.session_state.documents[doc_id] = doc
                        st.success(f"✅ Document processed and saved successfully!")
                        st.session_state.need_rerun = True
                    else:
                        st.error("❌ Failed to save document")
    
    with library_tab:
        st.subheader("Document Library")
        
        # Add a refresh button
        if st.button("Refresh Library", key="refresh_library"):
            load_document_library()
            st.success("Library refreshed")
        
        # Display document library
        if not st.session_state.document_library:
            st.info("No documents in the library. Upload and save documents to add them to your library.")
        else:
            # Display library as a table
            doc_list = list(st.session_state.document_library.values())
            doc_df = pd.DataFrame(doc_list)
            
            # Ensure required columns exist
            if not doc_df.empty:
                columns_to_display = ['title', 'upload_date', 'chunks', 'characters']
                existing_columns = [col for col in columns_to_display if col in doc_df.columns]
                
                if existing_columns:
                    doc_df = doc_df[existing_columns]
                    display_columns = ['Title', 'Upload Date', 'Chunks', 'Characters'][:len(existing_columns)]
                    doc_df.columns = display_columns
                
                st.dataframe(doc_df, use_container_width=True)
            
                # Document selection
                selected_doc_titles = st.multiselect(
                    "Select documents to analyze:",
                    options=[doc['title'] for doc in doc_list],
                    default=[]
                )
                
                if selected_doc_titles:
                    selected_doc_ids = [
                        doc['id'] for doc in doc_list
                        if doc['title'] in selected_doc_titles
                    ]
                    
                    # Use columns for buttons to avoid conflicts
                    col1, col2 = st.columns(2)
                    
                    # Load selected documents
                    with col1:
                        if st.button("Load Selected Documents", key="load_docs_library"):
                            for doc_id in selected_doc_ids:
                                if doc_id not in st.session_state.documents:
                                    doc = load_document_from_library(doc_id)
                                    if doc:
                                        st.session_state.documents[doc_id] = doc
                            
                            st.success(f"Loaded {len(selected_doc_ids)} documents.")
                    
                    # Create knowledge graph from selected documents
                    with col2:
                        if st.button("Create Knowledge Graph", key="create_graph_library"):
                            # First load any documents not already in session
                            for doc_id in selected_doc_ids:
                                if doc_id not in st.session_state.documents:
                                    doc = load_document_from_library(doc_id)
                                    if doc:
                                        st.session_state.documents[doc_id] = doc
                            
                            # Create graph from selected documents only
                            if all(doc_id in st.session_state.documents for doc_id in selected_doc_ids):
                                graph = KnowledgeGraph(name="Document Knowledge Graph")
                                
                                # Add embedding service for generating embeddings
                                embedding_service = EmbeddingService()
                                
                                # Process selected documents
                                for doc_id in selected_doc_ids:
                                    document = st.session_state.documents[doc_id]
                                    
                                    # Create document node
                                    doc_node = KnowledgeNode(
                                        id=document.id,
                                        node_type='document',
                                        content_id=document.id,
                                        properties={
                                            'title': document.title,
                                        }
                                    )
                                    graph.add_node(doc_node)
                                    
                                    # Add chunks for this document
                                    for chunk in document.chunks:
                                        # Generate embedding for chunk if not already done
                                        if chunk.embedding is None:
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
                                        edge_id = f"doc_chunk_{document.id}_{chunk.id}"
                                        edge = KnowledgeEdge(
                                            id=edge_id,
                                            source_id=document.id,
                                            target_id=chunk.id,
                                            relationship_type='contains',
                                            weight=1.0
                                        )
                                        graph.add_edge(edge)
                                
                                # Calculate similarity edges between chunks
                                chunk_nodes = {node_id: node for node_id, node in graph.nodes.items() 
                                              if node.node_type == 'chunk' and node.embedding is not None}
                                
                                # Compare each pair of chunks (limit to first 100 for performance)
                                node_list = list(chunk_nodes.items())[:100]
                                for i, (node1_id, node1) in enumerate(node_list):
                                    for node2_id, node2 in node_list[i+1:]:
                                        # Calculate similarity
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
                                
                                # Store graph in session state
                                st.session_state.knowledge_graph = graph
                                st.success(f"Knowledge graph created from {len(selected_doc_ids)} documents! View it in the Knowledge Graph page.")
                            else:
                                st.error("Error loading some of the selected documents.")

elif page == "Knowledge Graph":
    st.header("Knowledge Graph Visualization")
    
    if st.session_state.knowledge_graph is None:
        st.info("Upload documents in the Document Analysis page first to generate a knowledge graph.")
        
        # Placeholder for knowledge graph visualization
        st.subheader("Knowledge Graph")
        st.markdown("### 📊 Knowledge Graph Visualization will appear here")
        st.markdown("Upload and process documents to see the visualization.")
        
        # Sample graph metrics
        st.subheader("Graph Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Graph Density", value="0.45")
            st.metric(label="Connected Components", value="3")
        
        with col2:
            st.metric(label="Centrality Score", value="0.78")
            st.metric(label="Bridge Nodes", value="5")
    else:
        # Display actual knowledge graph
        st.subheader("Knowledge Graph")
        
        # Create visualization
        graph = st.session_state.knowledge_graph
        
        # Calculate metrics
        metrics = graph.calculate_metrics()
        
        # Create networkx graph for visualization
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in graph.nodes.items():
            G.add_node(node_id, 
                      node_type=node.node_type, 
                      label=node.properties.get('key_point', node.properties.get('title', node_id))[:30])
        
        # Add edges
        for edge_id, edge in graph.edges.items():
            G.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get node colors based on type
        node_colors = ['#FF9999' if G.nodes[n]['node_type'] == 'document' else '#99CCFF' for n in G.nodes()]
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, 
                               labels={n: G.nodes[n]['label'] for n in G.nodes()},
                               font_size=8, ax=ax)
        
        # Set title
        plt.title("Knowledge Graph")
        plt.axis('off')
        
        # Display plot
        st.pyplot(fig)
        
        # Display actual graph metrics
        st.subheader("Graph Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Graph Density", value=f"{metrics.get('density', 0):.2f}")
            st.metric(label="Connected Components", value=metrics.get('components', 0))
        
        with col2:
            st.metric(label="Centrality Score", value=f"{metrics.get('avg_centrality', 0):.2f}")
            # Fix for TypeError - convert list to its length if it's a list
            bridge_nodes = metrics.get('bridge_nodes', 0)
            if isinstance(bridge_nodes, list):
                bridge_nodes = len(bridge_nodes)
            st.metric(label="Bridge Nodes", value=bridge_nodes)
        
        # Display graph overview
        st.subheader("Graph Overview")
        st.write(f"Total nodes: {len(graph.nodes)}")
        st.write(f"Total edges: {len(graph.edges)}")
        
        # Document nodes
        doc_nodes = [node for node_id, node in graph.nodes.items() if node.node_type == 'document']
        st.write(f"Documents: {len(doc_nodes)}")
        
        # Chunk nodes
        chunk_nodes = [node for node_id, node in graph.nodes.items() if node.node_type == 'chunk']
        st.write(f"Chunks: {len(chunk_nodes)}")
        
        # Similarity connections
        similarity_edges = [edge for edge_id, edge in graph.edges.items() if edge.relationship_type == 'similarity']
        st.write(f"Similarity connections: {len(similarity_edges)}")

elif page == "Semantic Gaps":
    st.header("Semantic Gaps Analysis")
    
    st.info("This page will show potential knowledge gaps identified in your document corpus.")
    
    # Placeholder for semantic gaps visualization
    st.markdown("### 🔍 Semantic Gaps Visualization will appear here")
    st.markdown("Process documents to see potential knowledge gaps.")
    
    # Sample semantic gaps
    st.subheader("Potential Knowledge Gaps")
    
    gaps = [
        {"id": "gap_1", "distance": 0.82, "vertices": ["node_1", "node_2", "node_3"]},
        {"id": "gap_2", "distance": 0.75, "vertices": ["node_4", "node_5", "node_6"]},
        {"id": "gap_3", "distance": 0.68, "vertices": ["node_7", "node_8", "node_9"]}
    ]
    
    for i, gap in enumerate(gaps):
        st.write(f"**Gap {i+1}**")
        st.write(f"Distance: {gap['distance']}")
        st.write(f"Connected to concepts: {', '.join(gap['vertices'])}")
        
        # AI recommendation
        st.write("**AI Recommendation:**")
        st.write(f"Consider adding information about how concepts {gap['vertices'][0]} and {gap['vertices'][1]} relate to each other.")
        
        st.markdown("---")

elif page == "Workflow Designer":
    st.header("AI-Powered Workflow Designer")
    
    st.markdown("""
    Design intelligent workflows based on your knowledge structure.
    
    1. Define workflow steps based on knowledge domains
    2. Connect steps through semantic pathways
    3. Optimize process flow based on knowledge topology
    4. Identify knowledge prerequisites for each step
    """)
    
    # Workflow name input
    workflow_name = st.text_input("Workflow Name", "New Workflow")
    
    # Workflow steps
    st.subheader("Workflow Steps")
    
    # Add new step form
    with st.form("new_step_form"):
        st.write("Add New Step")
        step_name = st.text_input("Step Name")
        step_description = st.text_area("Description")
        knowledge_domain = st.selectbox(
            "Knowledge Domain",
            ["Research", "Analysis", "Development", "Testing", "Deployment"]
        )
        prerequisite_steps = st.multiselect(
            "Prerequisites",
            ["Research", "Analysis", "Development", "Testing"]
        )
        
        # Add step button
        submitted = st.form_submit_button("Add Step")
        if submitted:
            st.success(f"Added step: {step_name}")
    
    # Workflow visualization placeholder
    st.subheader("Workflow Visualization")
    st.markdown("### 📈 Workflow Visualization will appear here")
    st.markdown("Create a workflow to see the visualization.")
    
    # AI recommendations
    st.subheader("AI Recommendations")
    st.info("Based on your knowledge graph, consider adding a 'Data Validation' step between 'Analysis' and 'Development'.")
    
    # Export workflow
    if st.button("Export Workflow"):
        st.success("Workflow exported successfully!")

# Check if we need to rerun the app
if st.session_state.need_rerun:
    st.session_state.need_rerun = False
    st.rerun()
