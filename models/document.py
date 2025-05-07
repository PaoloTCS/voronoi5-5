import dataclasses
import uuid
from typing import Optional, List, Any, Dict
import numpy as np

@dataclasses.dataclass
class Document:
    """
    Represents a document with its content, metadata, and optional embedding.
    
    In the knowledge topology framework, documents serve as the primary sources of knowledge
    that are analyzed for semantic structure, relationships, and potential knowledge gaps.
    """
    title: str
    content: str
    id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    chunks: List[Any] = dataclasses.field(default_factory=list)
    
    # New fields for knowledge topology analysis
    knowledge_domain: Optional[str] = None  # Domain classification (e.g., "Physics", "Computer Science")
    knowledge_density: Optional[float] = None  # Measure of information density
    centrality_score: Optional[float] = None  # Importance in the knowledge graph
    
    def __post_init__(self):
        # Ensure embedding is None or a numpy array
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            try:
                self.embedding = np.array(self.embedding)
            except Exception as e:
                raise ValueError(f"Failed to convert embedding to numpy array: {e}")

    def __repr__(self) -> str:
        return f"Document(id={self.id}, title={self.title!r}, domain={self.knowledge_domain})"
    
    def get_summary(self) -> str:
        """Returns a summary of the document based on metadata and key statistics."""
        chunk_count = len(self.chunks)
        summary = f"Document: {self.title}\n"
        summary += f"ID: {self.id}\n"
        summary += f"Length: {len(self.content)} characters\n"
        summary += f"Chunks: {chunk_count}\n"
        
        if self.knowledge_domain:
            summary += f"Knowledge Domain: {self.knowledge_domain}\n"
        
        if self.knowledge_density is not None:
            summary += f"Knowledge Density: {self.knowledge_density:.4f}\n"
            
        if self.centrality_score is not None:
            summary += f"Centrality in Knowledge Graph: {self.centrality_score:.4f}\n"
            
        # Add metadata
        if self.metadata:
            summary += "Metadata:\n"
            for key, value in self.metadata.items():
                summary += f"  - {key}: {value}\n"
                
        return summary
    
    def calculate_knowledge_metrics(self) -> Dict[str, float]:
        """
        Calculate various knowledge metrics for this document.
        Returns a dict of metric names and values.
        """
        metrics = {}
        
        # Basic metrics
        metrics['word_count'] = len(self.content.split())
        metrics['chunk_count'] = len(self.chunks)
        
        # Knowledge density (can be refined with more sophisticated methods)
        # Simple baseline: unique words / total words
        words = self.content.lower().split()
        metrics['knowledge_density'] = len(set(words)) / max(1, len(words))
        self.knowledge_density = metrics['knowledge_density']
        
        return metrics
