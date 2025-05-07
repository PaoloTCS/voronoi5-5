import uuid
import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from models.knowledge_graph import KnowledgeGraph
from services.ai.gap_analyzer import GapAnalyzer

class WorkflowStep:
    """
    Represents a step in a knowledge-based workflow.
    
    Each step is associated with specific knowledge nodes from the graph
    and has prerequisites, outcomes, and execution details.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        knowledge_domain: str,
        knowledge_nodes: List[str] = None,
        prerequisites: List[str] = None,
        estimated_time: int = 0,
        difficulty: int = 1  # 1-5 scale
    ):
        self.id = id
        self.name = name
        self.description = description
        self.knowledge_domain = knowledge_domain
        self.knowledge_nodes = knowledge_nodes or []
        self.prerequisites = prerequisites or []
        self.estimated_time = estimated_time  # in minutes
        self.difficulty = difficulty
        self.outcomes = []
        self.resources = []
        self.completion_criteria = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the workflow step to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "knowledge_domain": self.knowledge_domain,
            "knowledge_nodes": self.knowledge_nodes,
            "prerequisites": self.prerequisites,
            "estimated_time": self.estimated_time,
            "difficulty": self.difficulty,
            "outcomes": self.outcomes,
            "resources": self.resources,
            "completion_criteria": self.completion_criteria
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowStep':
        """Create a workflow step from a dictionary."""
        step = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Unnamed Step"),
            description=data.get("description", ""),
            knowledge_domain=data.get("knowledge_domain", "Unknown"),
            knowledge_nodes=data.get("knowledge_nodes", []),
            prerequisites=data.get("prerequisites", []),
            estimated_time=data.get("estimated_time", 0),
            difficulty=data.get("difficulty", 1)
        )
        
        step.outcomes = data.get("outcomes", [])
        step.resources = data.get("resources", [])
        step.completion_criteria = data.get("completion_criteria", [])
        
        return step


class Workflow:
    """
    Represents a knowledge-based workflow.
    
    A workflow is a directed graph of steps, where edges represent
    dependencies between steps. The workflow is designed based on
    the knowledge topology to optimize learning and execution.
    """
    
    def __init__(self, name: str, description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.steps = {}  # Dict[step_id, WorkflowStep]
        self.dependencies = []  # List[(source_id, target_id)]
        self.graph = nx.DiGraph()  # Directed graph representation
        self.metrics = {}
        
    def add_step(self, step: WorkflowStep) -> str:
        """Add a step to the workflow and return its ID."""
        self.steps[step.id] = step
        
        # Add to graph
        self.graph.add_node(
            step.id,
            name=step.name,
            description=step.description,
            knowledge_domain=step.knowledge_domain,
            difficulty=step.difficulty
        )
        
        # Add dependencies from prerequisites
        for prereq_id in step.prerequisites:
            if prereq_id in self.steps:
                self.dependencies.append((prereq_id, step.id))
                self.graph.add_edge(prereq_id, step.id)
        
        return step.id
    
    def add_dependency(self, source_id: str, target_id: str) -> bool:
        """
        Add a dependency from source step to target step.
        
        Returns True if successful, False if it would create a cycle.
        """
        if source_id not in self.steps or target_id not in self.steps:
            return False
        
        # Check if adding this edge would create a cycle
        test_graph = self.graph.copy()
        test_graph.add_edge(source_id, target_id)
        
        if nx.is_directed_acyclic_graph(test_graph):
            self.dependencies.append((source_id, target_id))
            self.graph.add_edge(source_id, target_id)
            
            # Update prerequisites if not already present
            target_step = self.steps[target_id]
            if source_id not in target_step.prerequisites:
                target_step.prerequisites.append(source_id)
            
            return True
        else:
            return False
    
    def remove_step(self, step_id: str) -> bool:
        """
        Remove a step from the workflow.
        
        Returns True if successful, False if step not found.
        """
        if step_id not in self.steps:
            return False
        
        # Remove from steps dict
        del self.steps[step_id]
        
        # Remove from dependencies
        self.dependencies = [(src, tgt) for src, tgt in self.dependencies
                           if src != step_id and tgt != step_id]
        
        # Remove from graph
        self.graph.remove_node(step_id)
        
        # Update prerequisites in other steps
        for step in self.steps.values():
            if step_id in step.prerequisites:
                step.prerequisites.remove(step_id)
        
        return True
    
    def get_critical_path(self) -> List[str]:
        """Get the critical path through the workflow based on estimated time."""
        if not self.steps:
            return []
        
        # Convert graph to a weighted graph based on estimated time
        weighted_graph = self.graph.copy()
        
        for node in weighted_graph.nodes():
            if node in self.steps:
                weighted_graph.nodes[node]['weight'] = -self.steps[node].estimated_time
        
        # Find the longest path (most negative weight)
        # Start with nodes that have no predecessors
        start_nodes = [n for n in weighted_graph.nodes() if weighted_graph.in_degree(n) == 0]
        
        if not start_nodes:
            return []
        
        # End with nodes that have no successors
        end_nodes = [n for n in weighted_graph.nodes() if weighted_graph.out_degree(n) == 0]
        
        if not end_nodes:
            return []
        
        # Find longest path from any start to any end
        longest_path = []
        max_time = 0
        
        for start in start_nodes:
            for end in end_nodes:
                try:
                    path = nx.shortest_path(weighted_graph, start, end, weight='weight')
                    path_time = sum(self.steps[node].estimated_time for node in path)
                    
                    if path_time > max_time:
                        max_time = path_time
                        longest_path = path
                except nx.NetworkXNoPath:
                    continue
        
        return longest_path
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate various workflow metrics.
        
        Returns a dict of metric names and values.
        """
        metrics = {}
        
        # Basic metrics
        metrics['step_count'] = len(self.steps)
        metrics['dependency_count'] = len(self.dependencies)
        
        # Complexity metrics
        if self.graph.nodes():
            # Average number of dependencies per step
            metrics['avg_dependencies'] = len(self.dependencies) / len(self.steps)
            
            # Longest path length
            try:
                metrics['longest_path_length'] = nx.dag_longest_path_length(self.graph)
            except:
                metrics['longest_path_length'] = 0
            
            # Critical path
            critical_path = self.get_critical_path()
            metrics['critical_path'] = critical_path
            metrics['critical_path_length'] = len(critical_path)
            
            # Total estimated time
            metrics['total_estimated_time'] = sum(step.estimated_time for step in self.steps.values())
            
            # Critical path time
            metrics['critical_path_time'] = sum(self.steps[node].estimated_time for node in critical_path)
            
            # Average difficulty
            metrics['avg_difficulty'] = sum(step.difficulty for step in self.steps.values()) / len(self.steps)
        
        # Store metrics
        self.metrics = metrics
        
        return metrics
    
    def get_execution_plan(self) -> List[Dict[str, Any]]:
        """
        Generate a step-by-step execution plan for the workflow.
        
        Returns a list of steps in execution order.
        """
        if not self.steps:
            return []
        
        # Topological sort to get execution order
        try:
            execution_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            # Graph has cycles
            return []
        
        # Create execution plan
        execution_plan = []
        for step_id in execution_order:
            if step_id in self.steps:
                step = self.steps[step_id]
                
                # Get predecessors
                predecessors = list(self.graph.predecessors(step_id))
                
                # Create step plan
                step_plan = {
                    "id": step.id,
                    "name": step.name,
                    "description": step.description,
                    "knowledge_domain": step.knowledge_domain,
                    "estimated_time": step.estimated_time,
                    "difficulty": step.difficulty,
                    "prerequisites": [self.steps[p].name for p in predecessors if p in self.steps],
                    "outcomes": step.outcomes,
                    "resources": step.resources,
                    "completion_criteria": step.completion_criteria
                }
                
                execution_plan.append(step_plan)
        
        return execution_plan
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the workflow to a serializable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": {step_id: step.to_dict() for step_id, step in self.steps.items()},
            "dependencies": self.dependencies,
            "metrics": self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        """Create a workflow from a dictionary representation."""
        workflow = cls(
            name=data.get("name", "Unnamed Workflow"),
            description=data.get("description", "")
        )
        workflow.id = data.get("id", str(uuid.uuid4()))
        
        # Add steps
        for step_id, step_data in data.get("steps", {}).items():
            step = WorkflowStep.from_dict(step_data)
            workflow.steps[step_id] = step
            
            # Add to graph
            workflow.graph.add_node(
                step.id,
                name=step.name,
                description=step.description,
                knowledge_domain=step.knowledge_domain,
                difficulty=step.difficulty
            )
        
        # Add dependencies
        for source_id, target_id in data.get("dependencies", []):
            if source_id in workflow.steps and target_id in workflow.steps:
                workflow.dependencies.append((source_id, target_id))
                workflow.graph.add_edge(source_id, target_id)
        
        # Add metrics
        workflow.metrics = data.get("metrics", {})
        
        return workflow


class WorkflowDesigner:
    """
    Service for designing and optimizing knowledge-based workflows.
    
    This service uses the knowledge graph to design workflows that
    optimize the flow of knowledge and learning.
    """
    
    def __init__(self, gap_analyzer: Optional[GapAnalyzer] = None):
        """
        Initialize the workflow designer.
        
        Args:
            gap_analyzer: Service for analyzing knowledge gaps
        """
        self.gap_analyzer = gap_analyzer or GapAnalyzer()
    
    def create_workflow_from_graph(
        self,
        knowledge_graph: KnowledgeGraph,
        name: str,
        description: str = "",
        knowledge_domains: List[str] = None,
        max_steps: int = 10
    ) -> Workflow:
        """
        Create a workflow based on the knowledge graph.
        
        Args:
            knowledge_graph: The knowledge graph to use
            name: Name of the workflow
            description: Description of the workflow
            knowledge_domains: List of knowledge domains to include
            max_steps: Maximum number of steps in the workflow
            
        Returns:
            A new Workflow object
        """
        workflow = Workflow(name=name, description=description)
        
        # Detect clusters in the knowledge graph if not already done
        if not hasattr(knowledge_graph, 'clusters') or not knowledge_graph.clusters:
            knowledge_graph.detect_clusters()
        
        # Identify the most important nodes based on centrality
        # Calculate centrality if not already done
        if not hasattr(knowledge_graph, 'metrics') or not knowledge_graph.metrics:
            knowledge_graph.calculate_metrics()
        
        # Get centrality scores
        centrality = knowledge_graph.metrics.get('centrality', {})
        
        if not centrality:
            # Calculate centrality directly
            centrality = {
                'degree': nx.degree_centrality(knowledge_graph.graph),
                'betweenness': nx.betweenness_centrality(knowledge_graph.graph),
                'closeness': nx.closeness_centrality(knowledge_graph.graph)
            }
        
        # Combine centrality scores
        combined_centrality = {}
        for node_id in knowledge_graph.nodes:
            combined_score = (
                centrality.get('degree', {}).get(node_id, 0) +
                centrality.get('betweenness', {}).get(node_id, 0) +
                centrality.get('closeness', {}).get(node_id, 0)
            )
            combined_centrality[node_id] = combined_score
        
        # Sort nodes by combined centrality (descending)
        sorted_nodes = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
        
        # Filter by knowledge domain if specified
        filtered_nodes = []
        for node_id, score in sorted_nodes:
            node = knowledge_graph.nodes.get(node_id)
            if node:
                # Check if node's domain matches any of the requested domains
                domain = node.properties.get('knowledge_domain')
                if not knowledge_domains or not domain or domain in knowledge_domains:
                    filtered_nodes.append((node_id, score))
        
        # Limit to max_steps most important nodes
        important_nodes = filtered_nodes[:max_steps]
        
        # Create steps from important nodes
        steps_created = 0
        for node_id, score in important_nodes:
            node = knowledge_graph.nodes.get(node_id)
            if node:
                # Create a step name and description based on node type
                if node.node_type == 'document':
                    title = node.properties.get('title', f"Document {node.content_id}")
                    name = f"Study {title}"
                    description = f"Review the document '{title}' to understand its key concepts."
                elif node.node_type == 'chunk':
                    key_point = node.properties.get('key_point', f"Concept {node.content_id}")
                    name = f"Learn {key_point}"
                    description = f"Study the concept '{key_point}' and its implications."
                else:
                    name = f"Explore {node.node_type} {node.content_id}"
                    description = f"Investigate this {node.node_type} and its connections."
                
                # Get domain or use default
                domain = node.properties.get('knowledge_domain', "General Knowledge")
                
                # Estimate difficulty based on centrality (higher centrality = higher difficulty)
                difficulty = min(5, max(1, int(score * 10) + 1))
                
                # Estimate time based on node type and complexity
                if node.node_type == 'document':
                    # Estimate based on content length
                    content_length = len(node.properties.get('content', ''))
                    estimated_time = max(15, content_length // 1000)  # 1 minute per 1000 chars
                else:
                    # Default time for other node types
                    estimated_time = 30
                
                # Create a step
                step = WorkflowStep(
                    id=str(uuid.uuid4()),
                    name=name,
                    description=description,
                    knowledge_domain=domain,
                    knowledge_nodes=[node_id],
                    estimated_time=estimated_time,
                    difficulty=difficulty
                )
                
                # Add outcomes based on node properties
                if node.properties.get('key_point'):
                    step.outcomes.append(f"Understand: {node.properties.get('key_point')}")
                
                # Add completion criteria
                step.completion_criteria.append(f"Can explain the main concepts in {name}")
                
                # Add to workflow
                workflow.add_step(step)
                steps_created += 1
        
        # Add dependencies based on graph connections
        for step1_id, step1 in workflow.steps.items():
            for step2_id, step2 in workflow.steps.items():
                if step1_id != step2_id:
                    # Check if there's a connection in the knowledge graph
                    node1_id = step1.knowledge_nodes[0] if step1.knowledge_nodes else None
                    node2_id = step2.knowledge_nodes[0] if step2.knowledge_nodes else None
                    
                    if node1_id and node2_id:
                        # Check for direct connection
                        if knowledge_graph.graph.has_edge(node1_id, node2_id):
                            workflow.add_dependency(step1_id, step2_id)
                        elif knowledge_graph.graph.has_edge(node2_id, node1_id):
                            workflow.add_dependency(step2_id, step1_id)
                        else:
                            # Check for path
                            try:
                                path = nx.shortest_path(knowledge_graph.graph, node1_id, node2_id)
                                if len(path) <= 3:  # Only add if path is reasonably short
                                    workflow.add_dependency(step1_id, step2_id)
                            except nx.NetworkXNoPath:
                                pass
        
        # Calculate workflow metrics
        workflow.calculate_metrics()
        
        return workflow
    
    def optimize_workflow(self, workflow: Workflow, optimization_goal: str = "time") -> Workflow:
        """
        Optimize a workflow based on the specified goal.
        
        Args:
            workflow: The workflow to optimize
            optimization_goal: Goal to optimize for ('time', 'learning', 'complexity')
            
        Returns:
            An optimized workflow
        """
        # Create a copy of the workflow
        optimized = Workflow.from_dict(workflow.to_dict())
        
        if optimization_goal == "time":
            # Optimize for minimal time
            # Find redundant steps and merge or remove them
            redundant_steps = []
            
            for step1_id, step1 in optimized.steps.items():
                for step2_id, step2 in optimized.steps.items():
                    if step1_id != step2_id and step1_id not in redundant_steps and step2_id not in redundant_steps:
                        # Check if steps cover similar knowledge
                        nodes1 = set(step1.knowledge_nodes)
                        nodes2 = set(step2.knowledge_nodes)
                        
                        # If steps share many nodes, they might be redundant
                        if nodes1 and nodes2:
                            overlap = nodes1.intersection(nodes2)
                            if len(overlap) / min(len(nodes1), len(nodes2)) > 0.7:
                                # Steps are similar, mark the less important one as redundant
                                if step1.difficulty < step2.difficulty:
                                    redundant_steps.append(step1_id)
                                else:
                                    redundant_steps.append(step2_id)
            
            # Remove redundant steps
            for step_id in redundant_steps:
                optimized.remove_step(step_id)
            
        elif optimization_goal == "learning":
            # Optimize for learning effectiveness
            # Add intermediate steps for difficult transitions
            new_steps = []
            
            # Look for difficult transitions (big jumps in difficulty)
            for source_id, target_id in optimized.dependencies:
                source = optimized.steps.get(source_id)
                target = optimized.steps.get(target_id)
                
                if source and target and target.difficulty - source.difficulty > 2:
                    # This is a big jump in difficulty, add an intermediate step
                    intermediate_step = WorkflowStep(
                        id=str(uuid.uuid4()),
                        name=f"Bridge concepts from {source.name} to {target.name}",
                        description=f"This step helps connect the concepts in {source.name} to prepare for {target.name}.",
                        knowledge_domain=source.knowledge_domain,
                        prerequisites=[source_id],
                        estimated_time=20,
                        difficulty=source.difficulty + 1
                    )
                    
                    # Add outcomes
                    intermediate_step.outcomes.append("Better preparation for more advanced concepts")
                    
                    # Add to new steps list
                    new_steps.append((intermediate_step, source_id, target_id))
            
            # Add the new intermediate steps
            for step, source_id, target_id in new_steps:
                # Add step
                step_id = optimized.add_step(step)
                
                # Update dependencies
                optimized.remove_dependency(source_id, target_id)
                optimized.add_dependency(source_id, step_id)
                optimized.add_dependency(step_id, target_id)
            
        elif optimization_goal == "complexity":
            # Optimize for reduced complexity
            # Simplify the workflow by removing non-critical dependencies
            dependencies_to_remove = []
            
            # Calculate critical path
            critical_path = optimized.get_critical_path()
            critical_edges = set(zip(critical_path[:-1], critical_path[1:]))
            
            # Identify non-critical dependencies that can be removed
            for source_id, target_id in optimized.dependencies:
                if (source_id, target_id) not in critical_edges:
                    # Check if removing this dependency would significantly change the graph
                    # by checking if there's an alternative path
                    test_graph = optimized.graph.copy()
                    test_graph.remove_edge(source_id, target_id)
                    
                    try:
                        # Check if there's still a path from source to target
                        path = nx.shortest_path(test_graph, source_id, target_id)
                        if path:
                            # There's an alternative path, so this dependency is redundant
                            dependencies_to_remove.append((source_id, target_id))
                    except nx.NetworkXNoPath:
                        # No alternative path, keep this dependency
                        pass
            
            # Remove redundant dependencies
            for source_id, target_id in dependencies_to_remove:
                # Remove dependency
                try:
                    optimized.dependencies.remove((source_id, target_id))
                    optimized.graph.remove_edge(source_id, target_id)
                    
                    # Update prerequisites in target step
                    if target_id in optimized.steps and source_id in optimized.steps[target_id].prerequisites:
                        optimized.steps[target_id].prerequisites.remove(source_id)
                except ValueError:
                    pass
        
        # Recalculate workflow metrics
        optimized.calculate_metrics()
        
        return optimized
    
    def add_gap_filling_steps(self, workflow: Workflow, knowledge_graph: KnowledgeGraph) -> Workflow:
        """
        Add steps to address knowledge gaps in the workflow.
        
        Args:
            workflow: The workflow to enhance
            knowledge_graph: The knowledge graph to analyze for gaps
            
        Returns:
            An enhanced workflow with gap-filling steps
        """
        # Create a copy of the workflow
        enhanced = Workflow.from_dict(workflow.to_dict())
        
        # Get semantic gaps
        semantic_gaps = self.gap_analyzer.identify_semantic_gaps(knowledge_graph)
        
        # Get nodes in the current workflow
        workflow_nodes = set()
        for step in enhanced.steps.values():
            workflow_nodes.update(step.knowledge_nodes)
        
        # Add steps for significant gaps that aren't already covered
        for gap in semantic_gaps:
            vertices = gap.get('vertices', [])
            
            # Check if any vertices are already in the workflow
            if any(vertex in workflow_nodes for vertex in vertices):
                # Calculate centroid - we'll use this as the focus of the new step
                gap_centroid = gap.get('centroid')
                
                if gap_centroid:
                    # Create a gap-filling step
                    step = WorkflowStep(
                        id=str(uuid.uuid4()),
                        name=f"Address knowledge gap {gap.get('id', 'unknown')}",
                        description=f"This step addresses a gap in knowledge between related concepts.",
                        knowledge_domain="Knowledge Gap",
                        knowledge_nodes=vertices,
                        estimated_time=45,
                        difficulty=3
                    )
                    
                    # Add outcomes
                    step.outcomes.append("Fill a critical gap in knowledge")
                    step.outcomes.append("Connect related concepts more effectively")
                    
                    # Add completion criteria
                    step.completion_criteria.append("Can explain the relationship between the connected concepts")
                    
                    # Add to workflow
                    step_id = enhanced.add_step(step)
                    
                    # Add dependencies to vertices that are in the workflow
                    for vertex in vertices:
                        for existing_step_id, existing_step in enhanced.steps.items():
                            if vertex in existing_step.knowledge_nodes:
                                # Make the gap-filling step dependent on existing steps
                                enhanced.add_dependency(existing_step_id, step_id)
        
        # Recalculate workflow metrics
        enhanced.calculate_metrics()
        
        return enhanced
    
    def create_learning_path(
        self,
        knowledge_graph: KnowledgeGraph,
        start_concept: str,
        target_concept: str,
        name: str = "Learning Path"
    ) -> Workflow:
        """
        Create a learning path from start concept to target concept.
        
        Args:
            knowledge_graph: The knowledge graph to use
            start_concept: Starting concept description
            target_concept: Target concept description
            name: Name of the learning path
            
        Returns:
            A workflow representing the learning path
        """
        # Generate a knowledge path using the gap analyzer
        path_info = self.gap_analyzer.generate_knowledge_path(
            knowledge_graph, start_concept, target_concept
        )
        
        if not path_info.get('path_exists', False):
            # No path exists, create a minimal workflow
            workflow = Workflow(
                name=name,
                description=f"Learning path from '{start_concept}' to '{target_concept}'. {path_info.get('explanation', '')}"
            )
            return workflow
        
        # Create a workflow based on the path
        workflow = Workflow(
            name=name,
            description=f"Learning path from '{start_concept}' to '{target_concept}'"
        )
        
        # Get path nodes
        path = path_info.get('path', [])
        path_details = path_info.get('path_details', [])
        
        # Create a step for each node in the path
        previous_step_id = None
        for i, node_id in enumerate(path):
            node = knowledge_graph.nodes.get(node_id)
            if not node:
                continue
                
            # Get node details
            node_detail = path_details[i] if i < len(path_details) else {}
            
            # Create a step name and description based on node type
            if node.node_type == 'document':
                title = node.properties.get('title', f"Document {node.content_id}")
                name = f"Study {title}"
                description = f"Review the document '{title}' to understand its key concepts."
            elif node.node_type == 'chunk':
                key_point = node.properties.get('key_point', f"Concept {node.content_id}")
                name = f"Learn {key_point}"
                description = f"Study the concept '{key_point}' and its implications."
            else:
                name = f"Explore {node.node_type} {node.content_id}"
                description = f"Investigate this {node.node_type} and its connections."
            
            # Get domain or use default
            domain = node.properties.get('knowledge_domain', "General Knowledge")
            
            # Create prerequisites based on previous step
            prerequisites = [previous_step_id] if previous_step_id else []
            
            # Create a step
            step = WorkflowStep(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                knowledge_domain=domain,
                knowledge_nodes=[node_id],
                prerequisites=prerequisites,
                estimated_time=30,  # Default 30 minutes
                difficulty=min(5, i + 1)  # Difficulty increases along the path
            )
            
            # Add outcomes
            if i < len(path) - 1:
                next_relationship = node_detail.get('next_relationship', 'leads to')
                step.outcomes.append(f"This {next_relationship} the next concept in the learning path")
            
            # Add completion criteria
            step.completion_criteria.append(f"Can explain the main concepts in {name}")
            
            # Add to workflow
            step_id = workflow.add_step(step)
            
            # Update previous step for next iteration
            previous_step_id = step_id
        
        # Calculate workflow metrics
        workflow.calculate_metrics()
        
        return workflow
