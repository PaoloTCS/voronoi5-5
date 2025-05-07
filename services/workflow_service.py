import os
import json
from typing import List, Dict, Any, Optional, Union
from models.document import Document
from models.chunk import Chunk
from models.knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from services.ai.text_processor import TextProcessor
from services.ai.embedding_service import EmbeddingService
from services.ai.gap_analyzer import GapAnalyzer
from services.ai.workflow_designer import WorkflowDesigner, Workflow, WorkflowStep

class WorkflowService:
    """
    Main service for the AI-powered workflow application.
    
    This service integrates the different components of the system to provide
    a unified interface for document processing, knowledge graph construction,
    and workflow design.
    """
    
    def __init__(self, api_key: Optional[str] = None, data_dir: str = "data"):
        """
        Initialize the workflow service.
        
        Args:
            api_key: API key for AI services (e.g. OpenAI)
            data_dir: Directory for storing and loading data
        """
        self.data_dir = data_dir
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(data_dir, "documents"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "chunks"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "graphs"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "workflows"), exist_ok=True)
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.embedding_service = EmbeddingService(api_key=self.api_key)
        self.gap_analyzer = GapAnalyzer(embedding_service=self.embedding_service)
        self.workflow_designer = WorkflowDesigner(gap_analyzer=self.gap_analyzer)
        
        # Load data
        self.documents = {}
        self.chunks = {}
        self.graphs = {}
        self.workflows = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load data from disk."""
        # Load documents
        docs_dir = os.path.join(self.data_dir, "documents")
        for filename in os.listdir(docs_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(docs_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        doc = Document(
                            title=data.get("title", "Untitled"),
                            content=data.get("content", ""),
                            id=data.get("id")
                        )
                        doc.metadata = data.get("metadata", {})
                        self.documents[doc.id] = doc
                except Exception as e:
                    print(f"Error loading document {filename}: {e}")
        
        # Load chunks
        chunks_dir = os.path.join(self.data_dir, "chunks")
        for filename in os.listdir(chunks_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(chunks_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        chunk = Chunk(
                            id=data.get("id"),
                            document_id=data.get("document_id"),
                            content=data.get("content"),
                            importance_rank=data.get("importance_rank", 1),
                            key_point=data.get("key_point", ""),
                            context_label=data.get("context_label", ""),
                            start_char=data.get("start_char"),
                            end_char=data.get("end_char")
                        )
                        self.chunks[chunk.id] = chunk
                except Exception as e:
                    print(f"Error loading chunk {filename}: {e}")
        
        # Link chunks to documents
        for chunk in self.chunks.values():
            if chunk.document_id in self.documents:
                self.documents[chunk.document_id].chunks.append(chunk)
        
        # Load knowledge graphs
        graphs_dir = os.path.join(self.data_dir, "graphs")
        for filename in os.listdir(graphs_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(graphs_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        graph = KnowledgeGraph.from_dict(data)
                        self.graphs[graph.id] = graph
                except Exception as e:
                    print(f"Error loading graph {filename}: {e}")
        
        # Load workflows
        workflows_dir = os.path.join(self.data_dir, "workflows")
        for filename in os.listdir(workflows_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(workflows_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        workflow = Workflow.from_dict(data)
                        self.workflows[workflow.id] = workflow
                except Exception as e:
                    print(f"Error loading workflow {filename}: {e}")
    
    def _save_document(self, document: Document):
        """Save a document to disk."""
        filepath = os.path.join(self.data_dir, "documents", f"{document.id}.json")
        
        # Convert to dict
        data = {
            "id": document.id,
            "title": document.title,
            "content": document.content,
            "metadata": document.metadata,
            "knowledge_domain": document.knowledge_domain,
            "knowledge_density": document.knowledge_density,
            "centrality_score": document.centrality_score
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_chunk(self, chunk: Chunk):
        """Save a chunk to disk."""
        filepath = os.path.join(self.data_dir, "chunks", f"{chunk.id}.json")
        
        # Convert to dict
        data = {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "importance_rank": chunk.importance_rank,
            "key_point": chunk.key_point,
            "context_label": chunk.context_label,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "centrality_score": chunk.centrality_score,
            "semantic_uniqueness": chunk.semantic_uniqueness,
            "boundary_score": chunk.boundary_score,
            "bridge_score": chunk.bridge_score
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_graph(self, graph: KnowledgeGraph):
        """Save a knowledge graph to disk."""
        filepath = os.path.join(self.data_dir, "graphs", f"{graph.id}.json")
        
        # Convert to dict
        data = graph.to_dict()
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_workflow(self, workflow: Workflow):
        """Save a workflow to disk."""
        filepath = os.path.join(self.data_dir, "workflows", f"{workflow.id}.json")
        
        # Convert to dict
        data = workflow.to_dict()
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_document(self, title: str, content: str, metadata: Dict[str, Any] = None) -> Document:
        """
        Create a new document in the system.
        
        Args:
            title: Document title
            content: Document content
            metadata: Additional metadata
            
        Returns:
            The newly created Document object
        """
        # Create document
        document = Document(
            title=title,
            content=content,
            metadata=metadata or {}
        )
        
        # Add to documents dict
        self.documents[document.id] = document
        
        # Process document to extract chunks
        self.text_processor.extract_chunks_by_paragraphs(document)
        
        # Calculate document metrics
        document.calculate_knowledge_metrics()
        
        # Save document
        self._save_document(document)
        
        # Save chunks
        for chunk in document.chunks:
            self.chunks[chunk.id] = chunk
            self._save_chunk(chunk)
        
        return document
    
    def update_document(self, document_id: str, title: Optional[str] = None, 
                      content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[Document]:
        """
        Update an existing document.
        
        Args:
            document_id: ID of the document to update
            title: New title (optional)
            content: New content (optional)
            metadata: New metadata (optional)
            
        Returns:
            The updated Document object, or None if not found
        """
        if document_id not in self.documents:
            return None
        
        document = self.documents[document_id]
        
        # Update fields if provided
        if title is not None:
            document.title = title
        
        if content is not None:
            document.content = content
            
            # If content changed, re-process chunks
            old_chunks = document.chunks
            
            # Remove old chunks from global dict
            for chunk in old_chunks:
                if chunk.id in self.chunks:
                    del self.chunks[chunk.id]
            
            # Clear chunks and re-extract
            document.chunks = []
            self.text_processor.extract_chunks_by_paragraphs(document)
            
            # Add new chunks to global dict
            for chunk in document.chunks:
                self.chunks[chunk.id] = chunk
                self._save_chunk(chunk)
        
        if metadata is not None:
            document.metadata.update(metadata)
        
        # Recalculate document metrics
        document.calculate_knowledge_metrics()
        
        # Save document
        self._save_document(document)
        
        return document
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its chunks.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if successful, False if document not found
        """
        if document_id not in self.documents:
            return False
        
        document = self.documents[document_id]
        
        # Remove chunks
        for chunk in document.chunks:
            if chunk.id in self.chunks:
                del self.chunks[chunk.id]
                
                # Delete chunk file
                chunk_path = os.path.join(self.data_dir, "chunks", f"{chunk.id}.json")
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass
        
        # Remove document from dict
        del self.documents[document_id]
        
        # Delete document file
        doc_path = os.path.join(self.data_dir, "documents", f"{document_id}.json")
        try:
            os.remove(doc_path)
        except Exception:
            pass
        
        return True
    
    def create_knowledge_graph(self, name: str, document_ids: List[str] = None) -> KnowledgeGraph:
        """
        Create a knowledge graph from documents and/or chunks.
        
        Args:
            name: Name for the knowledge graph
            document_ids: IDs of documents to include
            
        Returns:
            The newly created KnowledgeGraph object
        """
        # Create new knowledge graph
        graph = KnowledgeGraph(name=name)
        
        # Add documents and chunks
        if document_ids:
            for doc_id in document_ids:
                if doc_id in self.documents:
                    document = self.documents[doc_id]
                    
                    # Create a document node
                    doc_node = KnowledgeNode(
                        id=str(doc_id),
                        node_type='document',
                        content_id=doc_id,
                        embedding=document.embedding,
                        properties={
                            'title': document.title,
                            'knowledge_domain': document.knowledge_domain,
                            'knowledge_density': document.knowledge_density
                        }
                    )
                    graph.add_node(doc_node)
                    
                    # Add chunks for this document
                    for chunk in document.chunks:
                        # Create a chunk node
                        chunk_node = KnowledgeNode(
                            id=str(chunk.id),
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
                        
                        # Create an edge from document to chunk
                        edge_id = f"doc_chunk_{document.id}_{chunk.id}"
                        edge = KnowledgeEdge(
                            id=edge_id,
                            source_id=document.id,
                            target_id=chunk.id,
                            relationship_type='contains',
                            weight=1.0
                        )
                        graph.add_edge(edge)
        
        # Add to graphs dict
        self.graphs[graph.id] = graph
        
        # Generate embeddings for nodes
        self._generate_embeddings_for_graph(graph)
        
        # Calculate similarity edges
        self._calculate_similarity_edges(graph)
        
        # Calculate graph metrics
        graph.calculate_metrics()
        
        # Detect clusters
        graph.detect_clusters()
        
        # Identify semantic gaps
        graph.identify_semantic_gaps()
        
        # Save graph
        self._save_graph(graph)
        
        return graph
    
    def update_knowledge_graph(self, graph_id: str, name: Optional[str] = None, 
                             add_document_ids: List[str] = None,
                             remove_document_ids: List[str] = None) -> Optional[KnowledgeGraph]:
        """
        Update an existing knowledge graph.
        
        Args:
            graph_id: ID of the graph to update
            name: New name (optional)
            add_document_ids: Document IDs to add
            remove_document_ids: Document IDs to remove
            
        Returns:
            The updated KnowledgeGraph object, or None if not found
        """
        if graph_id not in self.graphs:
            return None
        
        graph = self.graphs[graph_id]
        
        # Update name if provided
        if name is not None:
            graph.name = name
        
        # Add documents
        if add_document_ids:
            for doc_id in add_document_ids:
                if doc_id in self.documents:
                    document = self.documents[doc_id]
                    
                    # Skip if document already in graph
                    if doc_id in graph.nodes:
                        continue
                    
                    # Create a document node
                    doc_node = KnowledgeNode(
                        id=str(doc_id),
                        node_type='document',
                        content_id=doc_id,
                        embedding=document.embedding,
                        properties={
                            'title': document.title,
                            'knowledge_domain': document.knowledge_domain,
                            'knowledge_density': document.knowledge_density
                        }
                    )
                    graph.add_node(doc_node)
                    
                    # Add chunks for this document
                    for chunk in document.chunks:
                        # Create a chunk node
                        chunk_node = KnowledgeNode(
                            id=str(chunk.id),
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
                        
                        # Create an edge from document to chunk
                        edge_id = f"doc_chunk_{document.id}_{chunk.id}"
                        edge = KnowledgeEdge(
                            id=edge_id,
                            source_id=document.id,
                            target_id=chunk.id,
                            relationship_type='contains',
                            weight=1.0
                        )
                        graph.add_edge(edge)
        
        # Remove documents
        if remove_document_ids:
            for doc_id in remove_document_ids:
                if doc_id in graph.nodes:
                    # Get chunks for this document
                    chunk_ids = [node_id for node_id, node in graph.nodes.items()
                               if node.node_type == 'chunk' and node.properties.get('document_id') == doc_id]
                    
                    # Remove chunks
                    for chunk_id in chunk_ids:
                        graph.remove_node(chunk_id)
                    
                    # Remove document
                    graph.remove_node(doc_id)
        
        # Generate embeddings for new nodes
        self._generate_embeddings_for_graph(graph)
        
        # Calculate similarity edges
        self._calculate_similarity_edges(graph)
        
        # Recalculate graph metrics
        graph.calculate_metrics()
        
        # Detect clusters
        graph.detect_clusters()
        
        # Identify semantic gaps
        graph.identify_semantic_gaps()
        
        # Save graph
        self._save_graph(graph)
        
        return graph
    
    def delete_knowledge_graph(self, graph_id: str) -> bool:
        """
        Delete a knowledge graph.
        
        Args:
            graph_id: ID of the graph to delete
            
        Returns:
            True if successful, False if graph not found
        """
        if graph_id not in self.graphs:
            return False
        
        # Remove from dict
        del self.graphs[graph_id]
        
        # Delete file
        graph_path = os.path.join(self.data_dir, "graphs", f"{graph_id}.json")
        try:
            os.remove(graph_path)
        except Exception:
            pass
        
        return True
    
    def _generate_embeddings_for_graph(self, graph: KnowledgeGraph):
        """
        Generate embeddings for nodes in the graph that don't have them.
        
        Args:
            graph: Knowledge graph to process
        """
        for node_id, node in graph.nodes.items():
            if node.embedding is None:
                if node.node_type == 'document':
                    # Get document
                    document = self.documents.get(node.content_id)
                    if document:
                        # Generate embedding for document
                        embedding = self.embedding_service.get_embedding(document.content[:5000])  # Limit to first 5000 chars
                        node.embedding = embedding
                        document.embedding = embedding
                
                elif node.node_type == 'chunk':
                    # Get chunk
                    chunk = self.chunks.get(node.content_id)
                    if chunk:
                        # Generate embedding for chunk
                        embedding = self.embedding_service.get_embedding(chunk.content)
                        node.embedding = embedding
                        chunk.embedding = embedding
    
    def _calculate_similarity_edges(self, graph: KnowledgeGraph, min_similarity: float = 0.7):
        """
        Calculate similarity edges between nodes in the graph.
        
        Args:
            graph: Knowledge graph to process
            min_similarity: Minimum similarity threshold for creating edges
        """
        # Get nodes with embeddings
        embedding_nodes = {node_id: node for node_id, node in graph.nodes.items() if node.embedding is not None}
        
        # Calculate similarities between chunk nodes
        chunk_nodes = {node_id: node for node_id, node in embedding_nodes.items() if node.node_type == 'chunk'}
        
        # Compare each pair of chunks
        for node1_id, node1 in chunk_nodes.items():
            for node2_id, node2 in chunk_nodes.items():
                if node1_id != node2_id:
                    # Skip if already connected
                    if graph.graph.has_edge(node1_id, node2_id):
                        continue
                    
                    # Calculate similarity
                    similarity = np.dot(node1.embedding, node2.embedding) / (
                        np.linalg.norm(node1.embedding) * np.linalg.norm(node2.embedding)
                    )
                    
                    # Add edge if similarity is above threshold
                    if similarity >= min_similarity:
                        edge_id = f"sim_{node1_id}_{node2_id}"
                        edge = KnowledgeEdge(
                            id=edge_id,
                            source_id=node1_id,
                            target_id=node2_id,
                            relationship_type='similarity',
                            weight=float(similarity)
                        )
                        graph.add_edge(edge)
    
    def create_workflow(self, name: str, description: str, graph_id: str) -> Workflow:
        """
        Create a workflow based on a knowledge graph.
        
        Args:
            name: Name for the workflow
            description: Description of the workflow
            graph_id: ID of the knowledge graph to use
            
        Returns:
            The newly created Workflow object
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Knowledge graph {graph_id} not found")
        
        graph = self.graphs[graph_id]
        
        # Create workflow
        workflow = self.workflow_designer.create_workflow_from_graph(
            knowledge_graph=graph,
            name=name,
            description=description
        )
        
        # Add to workflows dict
        self.workflows[workflow.id] = workflow
        
        # Save workflow
        self._save_workflow(workflow)
        
        return workflow
    
    def optimize_workflow(self, workflow_id: str, optimization_goal: str = "time") -> Optional[Workflow]:
        """
        Optimize an existing workflow.
        
        Args:
            workflow_id: ID of the workflow to optimize
            optimization_goal: Goal to optimize for ('time', 'learning', 'complexity')
            
        Returns:
            The optimized Workflow object, or None if not found
        """
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        
        # Optimize workflow
        optimized = self.workflow_designer.optimize_workflow(
            workflow=workflow,
            optimization_goal=optimization_goal
        )
        
        # Update in dict
        self.workflows[optimized.id] = optimized
        
        # Save workflow
        self._save_workflow(optimized)
        
        return optimized
    
    def add_gap_filling_steps(self, workflow_id: str, graph_id: str) -> Optional[Workflow]:
        """
        Add steps to address knowledge gaps in the workflow.
        
        Args:
            workflow_id: ID of the workflow to enhance
            graph_id: ID of the knowledge graph to use
            
        Returns:
            The enhanced Workflow object, or None if not found
        """
        if workflow_id not in self.workflows or graph_id not in self.graphs:
            return None
        
        workflow = self.workflows[workflow_id]
        graph = self.graphs[graph_id]
        
        # Add gap-filling steps
        enhanced = self.workflow_designer.add_gap_filling_steps(
            workflow=workflow,
            knowledge_graph=graph
        )
        
        # Update in dict
        self.workflows[enhanced.id] = enhanced
        
        # Save workflow
        self._save_workflow(enhanced)
        
        return enhanced
    
    def create_learning_path(self, graph_id: str, start_concept: str, target_concept: str, name: str) -> Optional[Workflow]:
        """
        Create a learning path from start concept to target concept.
        
        Args:
            graph_id: ID of the knowledge graph to use
            start_concept: Starting concept description
            target_concept: Target concept description
            name: Name of the learning path
            
        Returns:
            A workflow representing the learning path, or None if graph not found
        """
        if graph_id not in self.graphs:
            return None
        
        graph = self.graphs[graph_id]
        
        # Create learning path
        workflow = self.workflow_designer.create_learning_path(
            knowledge_graph=graph,
            start_concept=start_concept,
            target_concept=target_concept,
            name=name
        )
        
        # Add to workflows dict
        self.workflows[workflow.id] = workflow
        
        # Save workflow
        self._save_workflow(workflow)
        
        return workflow
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow.
        
        Args:
            workflow_id: ID of the workflow to delete
            
        Returns:
            True if successful, False if workflow not found
        """
        if workflow_id not in self.workflows:
            return False
        
        # Remove from dict
        del self.workflows[workflow_id]
        
        # Delete file
        workflow_path = os.path.join(self.data_dir, "workflows", f"{workflow_id}.json")
        try:
            os.remove(workflow_path)
        except Exception:
            pass
        
        return True
    
    def analyze_gaps(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Analyze knowledge gaps in a graph.
        
        Args:
            graph_id: ID of the knowledge graph to analyze
            
        Returns:
            List of gap dictionaries, or empty list if graph not found
        """
        if graph_id not in self.graphs:
            return []
        
        graph = self.graphs[graph_id]
        
        # Identify semantic gaps
        gaps = self.gap_analyzer.identify_semantic_gaps(graph)
        
        # Get recommendations
        recommendations = self.gap_analyzer.recommend_new_content(graph)
        
        return {
            "gaps": gaps,
            "recommendations": recommendations
        }
