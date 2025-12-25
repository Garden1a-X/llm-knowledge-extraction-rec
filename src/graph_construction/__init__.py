"""
Graph Construction Module

Constructs three types of knowledge graphs:
1. Movie-side graph (movie -> attributes/knowledge points)
2. User-side graph (user -> interests/preferences)
3. User-movie interaction graph (with rating-based positive/negative edges)

Supports graph fusion and storage in various formats (NetworkX, PyG, Neo4j).
"""
