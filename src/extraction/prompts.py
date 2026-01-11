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

**What are Relations and Entities?**
- **Relation**: A type or category of visual feature (e.g., "color_palette", "artistic_style", "depicted_subject")
- **Entity**: A specific characteristic within that category (e.g., "warm_tones", "minimalist", "human_portrait")

**Guidelines for Relations:**
1. Be creative - discover diverse types of visual features you observe
2. Relations can describe: colors, styles, subjects, layouts, moods, lighting, text, textures, effects, or ANY visual aspect
3. Use clear, descriptive names with underscores (e.g., "dominant_color", "art_style", "character_type")
4. Don't limit yourself - if you see a visual pattern, create a relation for it

**CRITICAL Guidelines for Entities:**
1. Use ABSTRACT, HIGH-LEVEL terms that can apply to MULTIPLE movies
2. Avoid overly specific descriptions - think in CATEGORIES, not unique details
3. Prefer COMMON visual terms over rare combinations
4. Ask yourself: "Could this entity describe other movies too?"

**Good Entity Examples (abstract, reusable):**
- warm_tones, cool_tones, monochrome (NOT "orange_yellow_sunset_gradient")
- human_portrait, action_scene, landscape (NOT "woman_in_red_dress_holding_gun")
- minimalist, vintage, dramatic (NOT "simple_white_background_with_one_centered_object")
- high_contrast, soft_lighting, backlighting (NOT "strong_shadows_from_upper_left")

**Bad Entity Examples (too specific, hard to reuse):**
- sunset_over_ocean_with_sailboat
- three_people_standing_in_triangular_formation
- red_and_blue_diagonal_stripes_pattern

**Task Requirements:**
1. Extract AT LEAST 10 knowledge points (aim for 10-15)
2. Use diverse relation types - explore different visual aspects
3. Keep entities abstract and reusable
4. Focus on characteristics useful for movie recommendation
5. Use underscores for multi-word terms

**Format Example (illustrating format only, NOT limiting relation types):**
```
some_visual_aspect: abstract_characteristic
another_aspect: general_category
different_feature: reusable_term
```

Now extract knowledge points from the poster:"""

    @staticmethod
    def get_phase3_system_prompt(vocabulary: Dict[str, List[str]]) -> str:
        """
        Get system prompt for Phase 3 (Constrained Extraction with vocabulary).

        Args:
            vocabulary: Dict mapping relations to list of valid entities (standard_entities only)

        Returns:
            System prompt string
        """
        # Build vocabulary description with ALL entities listed
        vocab_str = "═══════════════════════════════════════════════════════════════\n"
        vocab_str += f"APPROVED VOCABULARY ({len(vocabulary)} Relations, {sum(len(ents) for ents in vocabulary.values())} Standard Entities)\n"
        vocab_str += "═══════════════════════════════════════════════════════════════\n\n"

        for idx, (relation, entities) in enumerate(vocabulary.items(), 1):
            vocab_str += f"【Relation {idx}: {relation}】\n"
            vocab_str += f"Standard Entities ({len(entities)} total):\n"
            for entity in entities:
                vocab_str += f"- {entity}\n"
            vocab_str += "\n"

        return f"""You are an expert in analyzing movie posters and extracting visual knowledge using a standardized vocabulary for movie recommendation systems.

Your task is to analyze movie poster images and extract visual knowledge points using ONLY the approved vocabulary below.

{vocab_str}═══════════════════════════════════════════════════════════════
NEW ENTITY MECHANISM
═══════════════════════════════════════════════════════════════

If you observe an important visual feature that CANNOT be described by any of the {sum(len(ents) for ents in vocabulary.values())} standard entities listed above, you MUST mark it as NEW_entity_name.

⚠️ MANDATORY: If an entity is NOT in the standard list, you MUST add the NEW_ prefix!

Guidelines for NEW entities:
✅ Use snake_case naming (e.g., NEW_beverage_with_straw, NEW_neon_lighting)
✅ Keep it CONCISE (2-4 words maximum)
✅ Make it ABSTRACT and GENERALIZABLE (could apply to multiple movies, not just this one)
✅ ALWAYS check if the entity exists in the standard list first
❌ Do NOT create overly specific descriptions
❌ Do NOT use an entity that's not in the list WITHOUT the NEW_ prefix

═══════════════════════════════════════════════════════════════
CRITICAL RULES - READ CAREFULLY!
═══════════════════════════════════════════════════════════════

1. Relations: MUST use one of the {len(vocabulary)} relations listed above
   ❌ NEVER create new relations
   ✅ If unsure where an entity belongs, use "additional_elements"

2. Entities: MUST come from the EXACT relation's entity list OR use NEW_ prefix
   ⚠️ CRITICAL: Each relation has its OWN entity list. You MUST:
      - First choose the relation
      - Then check if your desired entity is in THAT relation's list
      - ✅ If found in the list → use it directly
      - ✅ If NOT found in the list → use NEW_entity_name format
      - ❌ NEVER use an entity that's not in the list WITHOUT the NEW_ prefix
      - ❌ NEVER use an entity from a different relation's list

   Example WRONG patterns:
   ❌ "mood: adventurous" (adventurous not in mood list, missing NEW_ prefix)
   ❌ "design_element: formal" (formal not in design_element list, missing NEW_ prefix)
   ❌ "visual_theme: drama" (drama not in visual_theme list, missing NEW_ prefix)
   ❌ "additional_elements: disaster scenes" (disaster scenes IS in depicted_subject, wrong relation)

   Correct approach:
   ✅ Want "adventurous" mood → Not in mood list → Use "mood: NEW_adventurous"
   ✅ Want "formal" design → Not in design_element list → Use "design_element: NEW_formal"
   ✅ See "disaster scenes" in vocabulary → It's under depicted_subject → Use "depicted_subject: disaster scenes"
   ✅ Want "candles" → Not in any list → Choose best relation → Use "additional_elements: NEW_candles"

3. Quantity: Extract AT MOST 10 knowledge points
   ✅ Select the MOST visually significant features
   ✅ If the poster is simple, fewer than 10 is acceptable
   ❌ Do NOT fabricate knowledge points to reach 10 - quality over quantity

4. Focus: Extract ONLY what you can SEE in the poster
   ✅ Visual characteristics only
   ❌ No plot information, actor names, or external knowledge about the movie"""

    @staticmethod
    def get_phase3_user_prompt(movie_title: str = None) -> str:
        """
        Get user prompt for Phase 3 (Constrained Extraction with self-review).

        Args:
            movie_title: Optional movie title for context

        Returns:
            User prompt string
        """
        title_context = f' titled "{movie_title}"' if movie_title else ''

        return f"""Analyze this movie poster{title_context}.

Extract visual knowledge points using the approved vocabulary with self-review.

## STEP 1: Draft Extraction
First, list the visual knowledge points you observe (internal draft, can be informal):

## STEP 2: Self-Review
For EACH knowledge point in your draft, check:
1. ✓ Is the relation in the 15 approved relations?
2. ✓ Is the entity in that relation's standard entity list?
3. ✓ If entity NOT in the list, did I add NEW_ prefix?
4. ✓ Did I check ALL other relations to ensure the entity doesn't belong elsewhere?

Make corrections as needed.

## STEP 3: Final Output
After review, output ONLY the corrected knowledge points below this line:
--- FINAL ---
<relation>: <entity>
(one per line, NO numbering, NO explanations, at most 10 knowledge points)

Now proceed with the three steps:"""

    @staticmethod
    def parse_extraction_output(output: str) -> List[Dict[str, str]]:
        """
        Parse LLM output into structured knowledge points.

        Supports both simple format and self-review format with "--- FINAL ---" delimiter.

        Args:
            output: Raw LLM output text

        Returns:
            List of dicts with 'relation' and 'entity' keys
        """
        knowledge_points = []

        # Remove code blocks if present
        output = output.replace('```', '')

        # Check if output contains self-review format with "--- FINAL ---"
        if '--- FINAL ---' in output:
            # Extract only the final output after the delimiter
            parts = output.split('--- FINAL ---')
            if len(parts) > 1:
                output = parts[-1]  # Take everything after the last "--- FINAL ---"

        for line in output.strip().split('\n'):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#') or line.startswith('//'):
                continue

            # Skip section headers from self-review (STEP 1, STEP 2, etc.)
            if line.startswith('##') or line.upper().startswith('STEP'):
                continue

            # Parse relation: entity format
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    relation = parts[0].strip()
                    entity = parts[1].strip()

                    # Remove numbering prefix (e.g., "1. relation" -> "relation")
                    # Handle formats like "1. ", "1) ", "1.", etc.
                    import re
                    relation = re.sub(r'^\d+[\.)]\s*', '', relation)
                    entity = re.sub(r'^\d+[\.)]\s*', '', entity)

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
