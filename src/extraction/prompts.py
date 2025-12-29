"""
Prompt templates for knowledge extraction from movie posters.
"""

from typing import Dict, List
from pathlib import Path


class PromptTemplates:
    """Manages prompt templates for different extraction phases."""

    @staticmethod
    def get_phase1_system_prompt() -> str:
        """
        Get system prompt for Phase 1 (Free Exploration).

        Returns:
            System prompt string
        """
        return """You are an expert in analyzing movie posters and extracting visual knowledge points for movie recommendation systems.

Your task is to analyze movie poster images and extract visual knowledge in the form of relation-entity pairs.

IMPORTANT PRINCIPLES:
1. Extract ONLY what you can SEE in the poster - focus on visual characteristics
2. Knowledge points should be GENERALIZABLE - they should potentially apply to multiple movies, not be unique to just this one poster
3. Knowledge points should be RECOMMENDATION-RELEVANT - they should help distinguish movie preferences and styles
4. Do not include plot information, actor names, or external knowledge about the movie"""

    @staticmethod
    def get_phase1_user_prompt(movie_title: str = None) -> str:
        """
        Get user prompt for Phase 1 (Free Exploration).

        Args:
            movie_title: Optional movie title for context

        Returns:
            User prompt string
        """
        title_context = f' (Movie: "{movie_title}")' if movie_title else ''

        return f"""Analyze this movie poster{title_context} and extract visual knowledge points.

**Output Format:**
Provide knowledge points as relation-entity pairs, one per line:
```
<relation>: <entity>
```

**Relations to consider:**
- color_scheme: Overall color palette and tones
- visual_style: Aesthetic style (e.g., noir, minimalist, retro)
- main_element: Key visual objects or subjects
- composition: Layout and spatial arrangement
- mood: Emotional atmosphere conveyed
- lighting: Lighting characteristics
- typography: Text style (if prominent)
- texture: Surface quality or effects

**Guidelines:**
1. Extract AT LEAST 10 knowledge points (aim for 10-15)
2. Relations should be SINGLE WORDS or short phrases (e.g., "color_scheme", "visual_style")
3. Entities should be WORDS or SHORT PHRASES (e.g., "warm_tones", "cyberpunk_aesthetic")
4. Use underscores for multi-word terms (e.g., "warm_orange_tones")
5. Knowledge points should be GENERALIZABLE - potentially applicable to multiple movies
6. Focus on characteristics that are USEFUL FOR RECOMMENDATION

**Example:**
```
color_scheme: warm_orange_tones
color_scheme: deep_red_accents
visual_style: retro_aesthetic
visual_style: minimalist_design
main_element: silhouette_figure
main_element: urban_skyline
composition: centered_vertical_layout
mood: mysterious_atmosphere
mood: nostalgic_feeling
lighting: high_contrast
lighting: dramatic_shadows
typography: bold_sans_serif
```

Now extract knowledge points from the poster:"""

    @staticmethod
    def get_phase3_system_prompt(vocabulary: Dict[str, List[str]]) -> str:
        """
        Get system prompt for Phase 3 (Constrained Extraction with vocabulary).

        Args:
            vocabulary: Dict mapping relations to list of valid entities

        Returns:
            System prompt string
        """
        # Build vocabulary description
        vocab_str = "**Approved Vocabulary:**\n\n"
        for relation, entities in vocabulary.items():
            entities_str = ", ".join(entities[:10])
            if len(entities) > 10:
                entities_str += f", ... ({len(entities)} total)"
            vocab_str += f"- {relation}: {entities_str}\n"

        return f"""You are an expert in analyzing movie posters and extracting visual knowledge using a standardized vocabulary.

Your task is to analyze movie poster images and extract visual knowledge using ONLY the approved vocabulary below.

{vocab_str}

**Important:**
- Use ONLY relations and entities from the approved vocabulary
- If you see a visual feature that doesn't match any approved entity, use the "others" category for that relation
- Maintain consistency by using exact terms from the vocabulary
- Focus on what you can SEE in the poster"""

    @staticmethod
    def get_phase3_user_prompt(
        movie_title: str = None,
        relations: List[str] = None
    ) -> str:
        """
        Get user prompt for Phase 3 (Constrained Extraction).

        Args:
            movie_title: Optional movie title for context
            relations: List of relations to extract (if None, use all)

        Returns:
            User prompt string
        """
        title_context = f' (Movie: "{movie_title}")' if movie_title else ''
        relations_str = ""
        if relations:
            relations_str = "\n**Focus on these relations:**\n"
            relations_str += "\n".join(f"- {r}" for r in relations)

        return f"""Analyze this movie poster{title_context} and extract visual knowledge points using the approved vocabulary.
{relations_str}

**Output Format:**
Provide knowledge points as relation-entity pairs, one per line:
```
<relation>: <entity>
```

**Guidelines:**
1. Extract 5-12 knowledge points
2. Use ONLY entities from the approved vocabulary
3. Use "others" if no approved entity matches
4. Be precise and consistent

Now extract knowledge points from the poster:"""

    @staticmethod
    def parse_extraction_output(output: str) -> List[Dict[str, str]]:
        """
        Parse LLM output into structured knowledge points.

        Args:
            output: Raw LLM output text

        Returns:
            List of dicts with 'relation' and 'entity' keys
        """
        knowledge_points = []

        # Remove code blocks if present
        output = output.replace('```', '')

        for line in output.strip().split('\n'):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#') or line.startswith('//'):
                continue

            # Parse relation: entity format
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    relation = parts[0].strip()
                    entity = parts[1].strip()

                    # Basic validation
                    if relation and entity:
                        knowledge_points.append({
                            'relation': relation,
                            'entity': entity
                        })

        return knowledge_points

    @staticmethod
    def format_knowledge_points(
        knowledge_points: List[Dict[str, str]],
        format_type: str = 'text'
    ) -> str:
        """
        Format knowledge points for display or storage.

        Args:
            knowledge_points: List of knowledge point dicts
            format_type: 'text', 'json', or 'markdown'

        Returns:
            Formatted string
        """
        if format_type == 'text':
            return '\n'.join(
                f"{kp['relation']}: {kp['entity']}"
                for kp in knowledge_points
            )
        elif format_type == 'json':
            import json
            return json.dumps(knowledge_points, indent=2)
        elif format_type == 'markdown':
            lines = ["| Relation | Entity |", "|----------|--------|"]
            lines.extend(
                f"| {kp['relation']} | {kp['entity']} |"
                for kp in knowledge_points
            )
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unknown format_type: {format_type}")


# Example usage
if __name__ == '__main__':
    # Phase 1 example
    print("=" * 60)
    print("Phase 1: Free Exploration")
    print("=" * 60)
    print("\nSystem Prompt:")
    print(PromptTemplates.get_phase1_system_prompt())
    print("\nUser Prompt:")
    print(PromptTemplates.get_phase1_user_prompt("The Matrix"))

    # Parse example
    print("\n" + "=" * 60)
    print("Parse Example")
    print("=" * 60)
    example_output = """
    color_scheme: dark_blue_tones
    visual_style: cyberpunk_aesthetic
    main_element: digital_rain_effect
    mood: mysterious_atmosphere
    """
    parsed = PromptTemplates.parse_extraction_output(example_output)
    print(f"\nParsed {len(parsed)} knowledge points:")
    print(PromptTemplates.format_knowledge_points(parsed, 'markdown'))
