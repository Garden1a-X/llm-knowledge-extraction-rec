"""
Knowledge Graph Builder

Constructs three types of knowledge graphs:
1. Product-side graph (product -> attributes)
2. User-side graph (user -> interests)
3. User-item interaction graph (with positive/negative edges)
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple
from loguru import logger


class KnowledgeGraphBuilder:
    """Builds knowledge graphs from extracted knowledge points."""

    def __init__(self, backend: str = "networkx"):
        """
        Initialize the graph builder.

        Args:
            backend: Graph backend ("networkx", "neo4j")
        """
        self.backend = backend

        # Initialize graphs
        self.product_graph = None
        self.user_graph = None
        self.interaction_graph = None
        self.unified_graph = None

        self._init_graphs()

    def _init_graphs(self):
        """Initialize empty graphs."""
        if self.backend == "networkx":
            self.product_graph = nx.DiGraph()
            self.user_graph = nx.DiGraph()
            self.interaction_graph = nx.DiGraph()
            self.unified_graph = nx.DiGraph()
            logger.info("Initialized NetworkX graphs")

        elif self.backend == "neo4j":
            # TODO: Initialize Neo4j connection
            logger.info("Neo4j backend not yet implemented")
            raise NotImplementedError("Neo4j backend not yet implemented")

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def build_product_graph(
        self,
        products: List[Dict],
        product_knowledge: Dict[str, List[Dict]]
    ):
        """
        Build product-side knowledge graph.

        Args:
            products: List of product dictionaries
            product_knowledge: Dict mapping product_id to knowledge points
        """
        logger.info("Building product knowledge graph")

        for product in products:
            product_id = product['product_id']
            product_title = product.get('title', '')

            # Add product node
            self.product_graph.add_node(
                product_id,
                node_type='product',
                title=product_title
            )

            # Add knowledge point nodes and edges
            if product_id in product_knowledge:
                for kp in product_knowledge[product_id]:
                    kp_id = f"{kp['type']}:{kp['content']}"

                    # Add knowledge point node (if not exists)
                    if not self.product_graph.has_node(kp_id):
                        self.product_graph.add_node(
                            kp_id,
                            node_type='knowledge_point',
                            kp_type=kp['type'],
                            kp_content=kp['content']
                        )

                    # Add edge from product to knowledge point
                    self.product_graph.add_edge(
                        product_id,
                        kp_id,
                        edge_type='has_attribute',
                        confidence=kp.get('confidence', 1.0)
                    )

        logger.info(
            f"Product graph built: "
            f"{self.product_graph.number_of_nodes()} nodes, "
            f"{self.product_graph.number_of_edges()} edges"
        )

    def build_user_graph(
        self,
        users: List[str],
        user_interests: Dict[str, List[Dict]]
    ):
        """
        Build user-side interest graph.

        Args:
            users: List of user IDs
            user_interests: Dict mapping user_id to interest points
        """
        logger.info("Building user interest graph")

        for user_id in users:
            # Add user node
            self.user_graph.add_node(
                user_id,
                node_type='user'
            )

            # Add interest nodes and edges
            if user_id in user_interests:
                for interest in user_interests[user_id]:
                    interest_id = f"{interest['type']}:{interest['content']}"

                    # Add interest node (if not exists)
                    if not self.user_graph.has_node(interest_id):
                        self.user_graph.add_node(
                            interest_id,
                            node_type='interest',
                            interest_type=interest['type'],
                            interest_content=interest['content']
                        )

                    # Add edge from user to interest
                    self.user_graph.add_edge(
                        user_id,
                        interest_id,
                        edge_type='has_interest',
                        interest_type=interest['type'],  # long_term or short_term
                        confidence=interest.get('confidence', 1.0)
                    )

        logger.info(
            f"User graph built: "
            f"{self.user_graph.number_of_nodes()} nodes, "
            f"{self.user_graph.number_of_edges()} edges"
        )

    def build_interaction_graph(
        self,
        interactions: List[Dict]
    ):
        """
        Build user-item interaction graph with positive/negative edges.

        Args:
            interactions: List of interaction dictionaries
        """
        logger.info("Building user-item interaction graph")

        for interaction in interactions:
            user_id = interaction['user_id']
            product_id = interaction['product_id']
            rating = interaction.get('rating', 0)
            timestamp = interaction.get('timestamp', 0)

            # Add user and product nodes if not exist
            if not self.interaction_graph.has_node(user_id):
                self.interaction_graph.add_node(user_id, node_type='user')

            if not self.interaction_graph.has_node(product_id):
                self.interaction_graph.add_node(product_id, node_type='product')

            # Determine interaction type (positive/negative)
            interaction_type = self._classify_interaction(rating)

            # Add edge
            self.interaction_graph.add_edge(
                user_id,
                product_id,
                edge_type='interaction',
                interaction_type=interaction_type,
                rating=rating,
                timestamp=timestamp
            )

        logger.info(
            f"Interaction graph built: "
            f"{self.interaction_graph.number_of_nodes()} nodes, "
            f"{self.interaction_graph.number_of_edges()} edges"
        )

    def _classify_interaction(self, rating: float) -> str:
        """
        Classify interaction as positive/negative based on rating.

        Args:
            rating: Interaction rating

        Returns:
            "positive" or "negative"
        """
        # TODO: Adjust threshold based on dataset
        threshold = 3.5
        return "positive" if rating >= threshold else "negative"

    def build_unified_graph(self):
        """
        Build unified graph combining all three graphs.
        """
        logger.info("Building unified knowledge graph")

        # Combine all graphs
        self.unified_graph = nx.compose_all([
            self.product_graph,
            self.user_graph,
            self.interaction_graph
        ])

        logger.info(
            f"Unified graph built: "
            f"{self.unified_graph.number_of_nodes()} nodes, "
            f"{self.unified_graph.number_of_edges()} edges"
        )

    def query_product_subgraph(
        self,
        product_ids: List[str],
        max_depth: int = 2
    ) -> nx.DiGraph:
        """
        Query subgraph around specific products.

        Args:
            product_ids: List of product IDs
            max_depth: Maximum hop distance from product nodes

        Returns:
            Subgraph
        """
        # Get all nodes within max_depth hops
        nodes = set(product_ids)
        for product_id in product_ids:
            if self.product_graph.has_node(product_id):
                # BFS to get neighbors within max_depth
                for depth in range(max_depth):
                    neighbors = []
                    for node in list(nodes):
                        if self.product_graph.has_node(node):
                            neighbors.extend(self.product_graph.successors(node))
                            neighbors.extend(self.product_graph.predecessors(node))
                    nodes.update(neighbors)

        # Extract subgraph
        subgraph = self.product_graph.subgraph(nodes).copy()
        return subgraph

    def query_user_subgraph(
        self,
        user_id: str,
        max_depth: int = 2
    ) -> nx.DiGraph:
        """
        Query subgraph around a specific user.

        Args:
            user_id: User ID
            max_depth: Maximum hop distance from user node

        Returns:
            Subgraph
        """
        if not self.user_graph.has_node(user_id):
            return nx.DiGraph()

        # Get all nodes within max_depth hops
        nodes = {user_id}
        for depth in range(max_depth):
            neighbors = []
            for node in list(nodes):
                if self.user_graph.has_node(node):
                    neighbors.extend(self.user_graph.successors(node))
                    neighbors.extend(self.user_graph.predecessors(node))
            nodes.update(neighbors)

        # Extract subgraph
        subgraph = self.user_graph.subgraph(nodes).copy()
        return subgraph

    def save_graphs(self, output_dir: str):
        """
        Save all graphs to disk.

        Args:
            output_dir: Output directory path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save as GraphML format
        nx.write_graphml(
            self.product_graph,
            os.path.join(output_dir, "product_graph.graphml")
        )
        nx.write_graphml(
            self.user_graph,
            os.path.join(output_dir, "user_graph.graphml")
        )
        nx.write_graphml(
            self.interaction_graph,
            os.path.join(output_dir, "interaction_graph.graphml")
        )
        nx.write_graphml(
            self.unified_graph,
            os.path.join(output_dir, "unified_graph.graphml")
        )

        logger.info(f"Graphs saved to {output_dir}")

    def load_graphs(self, input_dir: str):
        """
        Load graphs from disk.

        Args:
            input_dir: Input directory path
        """
        import os

        self.product_graph = nx.read_graphml(
            os.path.join(input_dir, "product_graph.graphml")
        )
        self.user_graph = nx.read_graphml(
            os.path.join(input_dir, "user_graph.graphml")
        )
        self.interaction_graph = nx.read_graphml(
            os.path.join(input_dir, "interaction_graph.graphml")
        )
        self.unified_graph = nx.read_graphml(
            os.path.join(input_dir, "unified_graph.graphml")
        )

        logger.info(f"Graphs loaded from {input_dir}")
