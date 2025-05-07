import dataclasses
from typing import Optional, List, Dict, Any
import numpy as np
import uuid

@dataclasses.dataclass
class Chunk:
    """
    Represents a chunk of text extracted from a document.
    
    In the knowledge topology framework, chunks are the foundational units 
    that form the nodes in the knowledge graph. Each chunk represents a 
    coherent semantic unit that can be positioned in the N-dimensional 
    knowledge space based on its embedding.
    """
    id: str
    document_id: str
    content: str
    importance_rank: int
    key_point: str  # The main idea this chunk represents
    context_label: str  # e.g., chapter title, topic name
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    
    # New fields for knowledge topology analysis
    semantic_neighbors: List[str] = dataclasses.field(default_factory=list)  # IDs of semantically close chunks
    centrality_score: Optional[float] = None  # Centrality in the knowledge graph
    semantic_uniqueness: Optional[float] = None  # How unique this chunk is in the corpus
    boundary_score: Optional[float] = None  # Indicates if this chunk forms a conceptual boundary
    bridge_score: Optional[float] = None  # Indicates if this chunk connects different concept clusters
    knowledge_path_scores: Dict[str, float] = dataclasses.field(default_factory=dict)  # Scores for paths to other concepts
    
    # For Go analogy: territory metrics
    territory_size: Optional[float] = None  # Size of semantic territory influenced by this chunk
    territory_overlap: Dict[str, float] = dataclasses.field(default_factory=dict)  # Overlap with other territories

    def __repr__(self):
        return (
            f"Chunk(id={self.id!r}, document_id={self.document_id!r}, "
            f"importance_rank={self.importance_rank}, "
            f"key_point={self.key_point!r}, context_label={self.context_label!r}, "
            f"content='{self.content[:50]}...', "
            f"centrality={self.centrality_score}, uniqueness={self.semantic_uniqueness})"
        )
    
    def get_summary(self) -> str:
        """Returns a summary of the chunk with its key metrics."""
        summary = f"Chunk: {self.key_point}\n"
        summary += f"ID: {self.id}\n"
        summary += f"From Document: {self.document_id}\n"
        summary += f"Context: {self.context_label}\n"
        summary += f"Importance Rank: {self.importance_rank}\n"
        summary += f"Length: {len(self.content)} characters\n"
        
        if self.centrality_score is not None:
            summary += f"Centrality in Knowledge Graph: {self.centrality_score:.4f}\n"
            
        if self.semantic_uniqueness is not None:
            summary += f"Semantic Uniqueness: {self.semantic_uniqueness:.4f}\n"
            
        if self.boundary_score is not None:
            summary += f"Boundary Score: {self.boundary_score:.4f}\n"
            
        if self.bridge_score is not None:
            summary += f"Bridge Score: {self.bridge_score:.4f}\n"
            
        if self.semantic_neighbors:
            summary += f"Number of Semantic Neighbors: {len(self.semantic_neighbors)}\n"
            
        if self.territory_size is not None:
            summary += f"Territory Size: {self.territory_size:.4f}\n"
            
        # Preview of content
        summary += f"\nContent Preview: {self.content[:100]}...\n"
                
        return summary
    
    @classmethod
    def from_text(cls, text: str, document_id: str, context_label: str = "Unknown") -> 'Chunk':
        """Create a chunk directly from text with minimal metadata."""
        chunk_id = f"{document_id}_chunk_{str(uuid.uuid4())[:8]}"
        return cls(
            id=chunk_id,
            document_id=document_id,
            content=text,
            importance_rank=1,  # Default rank
            key_point=context_label,  # Use context_label as default key_point
            context_label=context_label
        )
