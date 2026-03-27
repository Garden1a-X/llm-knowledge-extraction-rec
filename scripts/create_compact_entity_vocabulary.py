#!/usr/bin/env python3
"""
Create compact entity vocabulary for Video Games (~180 entities total).

Goal: Entities should describe SHARED characteristics across items,
      enabling collaborative filtering and cross-item recommendations.

Strategy: Aggressive merging into high-level concepts.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import re


def normalize_entity(entity: str) -> str:
    """Basic normalization."""
    entity = entity.lower().strip()
    entity = re.sub(r'\s+', '_', entity)
    entity = re.sub(r'_+', '_', entity)
    entity = entity.strip('_')
    return entity


def map_to_compact_entity(entity: str, relation: str) -> str:
    """
    Map specific entity to compact, high-level concept.

    This enables entities to be shared across items.
    """
    entity = normalize_entity(entity)

    # Relation-specific mapping
    if relation == 'has_visual_style':
        return map_visual_style(entity)
    elif relation == 'has_color_palette':
        return map_color_palette(entity)
    elif relation == 'features_character':
        return map_character(entity)
    elif relation == 'set_in_environment':
        return map_environment(entity)
    elif relation == 'has_atmosphere':
        return map_atmosphere(entity)
    elif relation == 'features_weapon':
        return map_weapon(entity)
    elif relation == 'features_vehicle':
        return map_vehicle(entity)
    elif relation == 'features_creature':
        return map_creature(entity)
    elif relation == 'has_genre_indicator':
        return map_genre(entity)
    elif relation == 'shows_perspective':
        return map_perspective(entity)
    elif relation == 'shows_platform':
        return map_platform(entity)
    elif relation == 'has_graphics_quality':
        return map_graphics_quality(entity)
    elif relation == 'has_ui_elements':
        return map_ui_elements(entity)
    elif relation == 'has_text_element':
        return map_text_element(entity)
    elif relation == 'has_additional_property':
        return map_additional_property(entity)
    else:
        return entity


def map_visual_style(entity: str) -> str:
    """Map to ~8 visual styles."""
    if re.search(r'realistic.*3d|3d.*realistic', entity):
        return 'realistic_3d'
    elif re.search(r'^3d$|^3d_graphics$', entity):
        return '3d'
    elif re.search(r'cartoon|cel.*shad', entity):
        return 'cartoon'
    elif re.search(r'anime', entity):
        return 'anime'
    elif re.search(r'pixel|8.*bit|16.*bit|retro.*2d', entity):
        return 'pixel_art'
    elif re.search(r'realistic|photorealistic', entity):
        return 'realistic'
    elif re.search(r'abstract|minimalist|geometric', entity):
        return 'abstract'
    elif re.search(r'stylized|artistic|illustrated|hand.*drawn|painted', entity):
        return 'stylized'
    else:
        return 'other_style'


def map_color_palette(entity: str) -> str:
    """Map to ~12 color categories."""
    # Warm colors
    if re.search(r'orange|red|yellow|warm', entity) and not re.search(r'blue|green|cool', entity):
        if re.search(r'bright|vibrant|neon', entity):
            return 'bright_warm'
        elif re.search(r'dark|muted', entity):
            return 'dark_warm'
        else:
            return 'warm_colors'

    # Cool colors
    elif re.search(r'blue|cyan|purple|cool', entity) and not re.search(r'orange|red|yellow', entity):
        if re.search(r'bright|vibrant|neon', entity):
            return 'bright_cool'
        elif re.search(r'dark|muted', entity):
            return 'dark_cool'
        else:
            return 'cool_colors'

    # Green/nature
    elif re.search(r'green', entity):
        return 'green_nature'

    # Monochrome
    elif re.search(r'^black$|^white$|^gray|^grey|monochrome', entity):
        return 'monochrome'

    # Dark palette
    elif re.search(r'dark|black', entity):
        return 'dark_palette'

    # Bright/vibrant
    elif re.search(r'bright|vibrant|colorful|vivid|neon|saturated', entity):
        return 'bright_vibrant'

    # Muted/pastel
    elif re.search(r'muted|pastel|soft|subdued', entity):
        return 'muted_pastel'

    # Mixed/multicolor
    elif re.search(r'and|multicolor|rainbow|varied', entity):
        return 'multicolor'

    else:
        return 'neutral_colors'


def map_character(entity: str) -> str:
    """Map to ~25 character types (not specific names)."""

    # Military/soldiers
    if re.search(r'soldier|military|marine|commando|spec.*ops|sniper|trooper', entity):
        return 'military_soldier'

    # Fantasy characters
    elif re.search(r'knight|warrior|wizard|mage|elf|dwarf|orc|barbarian|paladin', entity):
        return 'fantasy_warrior'

    # Sci-fi characters
    elif re.search(r'space.*marine|cyborg|robot|android|alien|astronaut', entity):
        return 'scifi_character'

    # Sports athletes
    elif re.search(r'basketball|football|soccer|baseball|athlete|player.*sport|sport.*player', entity):
        return 'sports_athlete'

    # Racing drivers
    elif re.search(r'race.*driver|driver.*race|racer', entity):
        return 'race_driver'

    # Armored characters
    elif re.search(r'armor|armored|suit.*armor', entity):
        return 'armored_character'

    # Ninjas/assassins
    elif re.search(r'ninja|assassin|stealth', entity):
        return 'stealth_character'

    # Animals (anthropomorphic)
    elif re.search(r'mario|sonic|yoshi|donkey.*kong|animal.*character|anthropomorphic', entity):
        return 'animal_character'

    # Superheroes
    elif re.search(r'superhero|hero.*suit|caped', entity):
        return 'superhero'

    # Monsters/enemies
    elif re.search(r'monster|zombie|demon|undead|enemy', entity):
        return 'monster_enemy'

    # Civilians
    elif re.search(r'civilian|casual|everyday|pedestrian', entity):
        return 'civilian'

    # Female characters
    elif re.search(r'female|woman|girl|princess|heroine', entity):
        return 'female_character'

    # Male characters (generic)
    elif re.search(r'male|man|boy|^guy$', entity):
        return 'male_character'

    # Cartoon characters
    elif re.search(r'cartoon|animated|stylized.*character', entity):
        return 'cartoon_character'

    # Multiple characters
    elif re.search(r'multiple|group|team|characters|various', entity):
        return 'multiple_characters'

    else:
        return 'character_other'


def map_environment(entity: str) -> str:
    """Map to ~18 environment types."""

    # Urban
    if re.search(r'urban|city|street|downtown|metropolis', entity):
        return 'urban'

    # Fantasy
    elif re.search(r'fantasy|medieval|castle|dungeon|magical|enchanted', entity):
        return 'fantasy'

    # Sci-fi/space
    elif re.search(r'sci.*fi|space|futuristic|cyberpunk|alien.*world|space.*station', entity):
        return 'scifi_space'

    # Military/war
    elif re.search(r'military|war|battlefield|combat.*zone|bunker|trench', entity):
        return 'military_warzone'

    # Nature/outdoor
    elif re.search(r'nature|forest|jungle|mountain|outdoor|wilderness|countryside', entity):
        return 'nature_outdoor'

    # Sports venues
    elif re.search(r'stadium|arena|court|field|track|sports.*venue', entity):
        return 'sports_venue'

    # Racing tracks
    elif re.search(r'race.*track|racing.*circuit|speedway', entity):
        return 'racing_track'

    # Ocean/underwater
    elif re.search(r'ocean|underwater|sea|aquatic', entity):
        return 'ocean_underwater'

    # Desert
    elif re.search(r'desert|arid|wasteland', entity):
        return 'desert'

    # Snow/ice
    elif re.search(r'snow|ice|arctic|frozen|winter', entity):
        return 'snow_ice'

    # Underground
    elif re.search(r'underground|cave|mine|tunnel', entity):
        return 'underground'

    # Indoor/interior
    elif re.search(r'indoor|interior|building|room|house', entity):
        return 'indoor'

    # Post-apocalyptic
    elif re.search(r'post.*apocal|dystopian|ruins|destroyed', entity):
        return 'post_apocalyptic'

    # Historical
    elif re.search(r'historical|ancient|classical|historical.*period', entity):
        return 'historical'

    # Abstract/minimal
    elif re.search(r'abstract|minimal|void|empty|plain', entity):
        return 'abstract_minimal'

    # Mixed/various
    elif re.search(r'various|multiple|mixed|different', entity):
        return 'varied_environments'

    else:
        return 'environment_other'


def map_atmosphere(entity: str) -> str:
    """Map to ~12 atmosphere types."""

    if re.search(r'dark.*gritty|gritty.*dark', entity):
        return 'dark_gritty'
    elif re.search(r'dark|ominous|foreboding|sinister', entity):
        return 'dark'
    elif re.search(r'gritty|harsh|brutal', entity):
        return 'gritty'
    elif re.search(r'whimsical|playful|lighthearted|cheerful|fun', entity):
        return 'whimsical'
    elif re.search(r'intense|action.*packed|adrenaline|exciting', entity):
        return 'intense_action'
    elif re.search(r'mysterious|enigmatic|suspenseful', entity):
        return 'mysterious'
    elif re.search(r'tense|suspense|nerve.*wracking', entity):
        return 'tense'
    elif re.search(r'epic|grand|cinematic|dramatic', entity):
        return 'epic'
    elif re.search(r'colorful|vibrant|lively|energetic', entity):
        return 'vibrant'
    elif re.search(r'calm|peaceful|serene|tranquil', entity):
        return 'calm'
    elif re.search(r'horror|scary|frightening|creepy', entity):
        return 'horror'
    else:
        return 'atmosphere_other'


def map_weapon(entity: str) -> str:
    """Map to ~10 weapon types."""

    if re.search(r'firearm|gun|rifle|pistol|assault.*rifle|smg|sniper', entity):
        return 'firearms'
    elif re.search(r'sword|blade|katana|saber', entity):
        return 'swords'
    elif re.search(r'melee|axe|hammer|mace|club|staff', entity):
        return 'melee_weapons'
    elif re.search(r'futuristic|laser|plasma|energy|sci.*fi', entity):
        return 'futuristic_weapons'
    elif re.search(r'bow|arrow|crossbow', entity):
        return 'ranged_projectile'
    elif re.search(r'explosive|grenade|rocket|missile|launcher', entity):
        return 'explosives'
    elif re.search(r'magic|spell|wand|staff.*magic', entity):
        return 'magical_weapons'
    elif re.search(r'heavy|artillery|cannon|turret', entity):
        return 'heavy_weapons'
    elif re.search(r'various|multiple|different', entity):
        return 'various_weapons'
    else:
        return 'weapon_other'


def map_vehicle(entity: str) -> str:
    """Map to ~12 vehicle types."""

    if re.search(r'race.*car|racing.*car|formula|sports.*car', entity):
        return 'race_car'
    elif re.search(r'car|automobile|sedan|vehicle.*road', entity):
        return 'car'
    elif re.search(r'tank|armored.*vehicle|apc', entity):
        return 'tank'
    elif re.search(r'military.*helicopter|attack.*helicopter|gunship', entity):
        return 'military_helicopter'
    elif re.search(r'helicopter|chopper', entity):
        return 'helicopter'
    elif re.search(r'military.*aircraft|fighter.*jet|bomber', entity):
        return 'military_aircraft'
    elif re.search(r'aircraft|plane|airplane|jet', entity):
        return 'aircraft'
    elif re.search(r'spacecraft|spaceship|starship|space.*fighter|starfighter', entity):
        return 'spacecraft'
    elif re.search(r'motorcycle|bike|motorbike', entity):
        return 'motorcycle'
    elif re.search(r'ship|boat|naval|vessel', entity):
        return 'watercraft'
    elif re.search(r'mech|robot.*vehicle|walker', entity):
        return 'mech'
    else:
        return 'vehicle_other'


def map_creature(entity: str) -> str:
    """Map to ~8 creature types."""

    if re.search(r'monster|demon|beast|fiend', entity):
        return 'monsters'
    elif re.search(r'dragon', entity):
        return 'dragons'
    elif re.search(r'zombie|undead|skeleton', entity):
        return 'undead'
    elif re.search(r'fantasy.*creature|mythical|legendary|griffin|phoenix', entity):
        return 'mythical_creatures'
    elif re.search(r'animal|wildlife|deer|fish|bird', entity):
        return 'animals'
    elif re.search(r'alien|extraterrestrial', entity):
        return 'aliens'
    elif re.search(r'insect|spider|bug', entity):
        return 'insects'
    else:
        return 'creature_other'


def map_genre(entity: str) -> str:
    """Map to ~12 genre indicators."""

    if re.search(r'fps|first.*person.*shooter', entity):
        return 'fps'
    elif re.search(r'rpg|role.*playing', entity):
        return 'rpg'
    elif re.search(r'action.*rpg', entity):
        return 'action_rpg'
    elif re.search(r'racing', entity):
        return 'racing'
    elif re.search(r'sports', entity):
        return 'sports'
    elif re.search(r'fighting', entity):
        return 'fighting'
    elif re.search(r'strategy|rts|real.*time.*strategy|turn.*based', entity):
        return 'strategy'
    elif re.search(r'action.*adventure|adventure', entity):
        return 'action_adventure'
    elif re.search(r'platformer|platform', entity):
        return 'platformer'
    elif re.search(r'puzzle', entity):
        return 'puzzle'
    elif re.search(r'horror|survival.*horror', entity):
        return 'horror'
    else:
        return 'genre_other'


def map_perspective(entity: str) -> str:
    """Map to 7 perspectives - already good."""

    if re.search(r'first.*person', entity):
        return 'first_person'
    elif re.search(r'third.*person', entity):
        return 'third_person'
    elif re.search(r'top.*down', entity):
        return 'top_down'
    elif re.search(r'side.*scroll', entity):
        return 'side_scrolling'
    elif re.search(r'isometric', entity):
        return 'isometric'
    elif re.search(r'overhead', entity):
        return 'overhead'
    elif re.search(r'2.5d', entity):
        return '2.5d'
    else:
        return 'perspective_other'


def map_platform(entity: str) -> str:
    """Map to ~15 major platforms."""

    if re.search(r'playstation.*5|ps5', entity):
        return 'playstation_5'
    elif re.search(r'playstation.*4|ps4', entity):
        return 'playstation_4'
    elif re.search(r'playstation.*3|ps3', entity):
        return 'playstation_3'
    elif re.search(r'playstation.*2|ps2', entity):
        return 'playstation_2'
    elif re.search(r'playstation|ps1|psx', entity):
        return 'playstation'
    elif re.search(r'xbox.*series', entity):
        return 'xbox_series'
    elif re.search(r'xbox.*one', entity):
        return 'xbox_one'
    elif re.search(r'xbox.*360', entity):
        return 'xbox_360'
    elif re.search(r'^xbox$', entity):
        return 'xbox'
    elif re.search(r'nintendo.*switch|^switch$', entity):
        return 'nintendo_switch'
    elif re.search(r'nintendo.*wii.*u', entity):
        return 'wii_u'
    elif re.search(r'nintendo.*wii|^wii$', entity):
        return 'wii'
    elif re.search(r'nintendo.*ds|3ds|^ds$', entity):
        return 'nintendo_ds'
    elif re.search(r'game.*boy|gameboy', entity):
        return 'game_boy'
    elif re.search(r'^pc$|windows|steam', entity):
        return 'pc'
    elif re.search(r'mobile|ios|android', entity):
        return 'mobile'
    elif re.search(r'retro|sega|saturn|dreamcast|genesis', entity):
        return 'retro_console'
    elif re.search(r'accessory|controller|peripheral', entity):
        return 'accessory'
    else:
        return 'platform_other'


def map_graphics_quality(entity: str) -> str:
    """Map to ~5 quality levels."""

    if re.search(r'high.*fidelity|high.*quality|high.*definition|4k|hd', entity):
        return 'high_fidelity'
    elif re.search(r'stylized|artistic', entity):
        return 'stylized'
    elif re.search(r'low.*fidelity|low.*quality|low.*poly|retro', entity):
        return 'low_fidelity'
    elif re.search(r'simple|basic|minimal', entity):
        return 'simple'
    else:
        return 'quality_other'


def map_ui_elements(entity: str) -> str:
    """Map to ~10 UI element types."""

    if re.search(r'hud|heads.*up|interface.*overlay', entity):
        return 'hud'
    elif re.search(r'menu|navigation', entity):
        return 'menu'
    elif re.search(r'score|scoreboard|points', entity):
        return 'scoreboard'
    elif re.search(r'health.*bar|stamina|energy.*bar', entity):
        return 'status_bars'
    elif re.search(r'minimap|radar|map', entity):
        return 'minimap'
    elif re.search(r'logo|branding|title.*logo', entity):
        return 'logo'
    elif re.search(r'button|prompt|command', entity):
        return 'button_prompts'
    elif re.search(r'icon|symbol', entity):
        return 'icons'
    elif re.search(r'minimal|clean|simple', entity):
        return 'minimal_ui'
    else:
        return 'ui_other'


def map_text_element(entity: str) -> str:
    """Map to ~8 text types."""

    if re.search(r'title.*logo|logo.*title|game.*logo', entity):
        return 'game_logo'
    elif re.search(r'title|game.*title', entity):
        return 'game_title'
    elif re.search(r'logo|branding', entity):
        return 'branding'
    elif re.search(r'description|tagline|slogan', entity):
        return 'description'
    elif re.search(r'instruction|tutorial|help', entity):
        return 'instructions'
    elif re.search(r'rating|esrb|pegi', entity):
        return 'rating'
    elif re.search(r'copyright|legal|trademark', entity):
        return 'legal_text'
    else:
        return 'text_other'


def map_additional_property(entity: str) -> str:
    """Map to ~15 additional properties."""

    if re.search(r'abstract.*background|background.*abstract', entity):
        return 'abstract_background'
    elif re.search(r'dynamic.*line|line.*dynamic|motion.*line', entity):
        return 'dynamic_lines'
    elif re.search(r'controller|gamepad|joystick', entity):
        return 'game_controller'
    elif re.search(r'mouse|gaming.*mouse', entity):
        return 'gaming_mouse'
    elif re.search(r'headset|headphone', entity):
        return 'gaming_headset'
    elif re.search(r'keyboard', entity):
        return 'gaming_keyboard'
    elif re.search(r'cartridge|disc|box.*art|cover.*art', entity):
        return 'physical_media'
    elif re.search(r'particle.*effect|special.*effect|visual.*effect', entity):
        return 'visual_effects'
    elif re.search(r'lighting|shadow|illumination', entity):
        return 'lighting_effects'
    elif re.search(r'camera.*angle|camera.*effect', entity):
        return 'camera_effects'
    elif re.search(r'texture|material', entity):
        return 'textures'
    elif re.search(r'animation|animated', entity):
        return 'animation'
    elif re.search(r'blur|focus|depth.*field', entity):
        return 'depth_effects'
    elif re.search(r'screenshot|gameplay.*image', entity):
        return 'screenshot'
    else:
        return 'property_other'


def create_compact_vocabulary(phase2_file: Path) -> dict:
    """Create compact vocabulary by aggressive entity merging."""

    print("="*70)
    print("Creating Compact Entity Vocabulary (~180 entities)")
    print("="*70)
    print()

    # Load Phase 2 results
    print(f"Loading: {phase2_file}")
    with open(phase2_file, 'r') as f:
        data = json.load(f)

    results = [r for r in data['results'] if r['status'] == 'success']
    print(f"  ✓ {len(results)} successful extractions")
    print()

    # Collect and map entities
    relation_entities = defaultdict(list)

    for result in results:
        for kp in result.get('knowledge_points', []):
            relation = kp['relation']
            entity = kp['entity']

            # Fix typo
            if relation == 'show_platform':
                relation = 'shows_platform'

            # Map to compact entity
            compact_entity = map_to_compact_entity(entity, relation)
            relation_entities[relation].append(compact_entity)

    # Count entities per relation
    vocabulary = {
        'dataset': 'amazon-videogames',
        'phase': 'phase2_constrained_compact',
        'vocabulary_version': '4.0',
        'description': 'Compact entity vocabulary with ~180 entities for collaborative filtering',
        'relations': {}
    }

    total_unique = 0

    for relation in sorted(relation_entities.keys()):
        entities = relation_entities[relation]
        counter = Counter(entities)

        entity_list = [
            {'entity': entity, 'count': count}
            for entity, count in counter.most_common()
        ]

        vocabulary['relations'][relation] = {
            'total_entities': len(entities),
            'unique_entities': len(counter),
            'entities': entity_list
        }

        total_unique += len(counter)

    vocabulary['total_unique_entities'] = total_unique

    return vocabulary


def print_vocabulary_stats(vocab: dict):
    """Print vocabulary statistics."""

    print("\n" + "="*70)
    print("Compact Vocabulary Statistics")
    print("="*70)
    print()

    print(f"Total unique entities across all relations: {vocab['total_unique_entities']}")
    print()

    for relation, data in sorted(vocab['relations'].items()):
        print(f"{relation}:")
        print(f"  Unique entities: {data['unique_entities']}")
        print(f"  Total occurrences: {data['total_entities']}")
        print(f"  Top 5:")
        for i, item in enumerate(data['entities'][:5], 1):
            print(f"    {i}. {item['entity']:<30} ({item['count']:>4})")
        print()


def main():
    phase2_file = Path("results/videogames/phase2_constrained_5pct.json")
    output_file = Path("results/videogames/entity_vocabulary_compact.json")

    # Create compact vocabulary
    vocab = create_compact_vocabulary(phase2_file)

    # Print stats
    print_vocabulary_stats(vocab)

    # Save
    with open(output_file, 'w') as f:
        json.dump(vocab, f, indent=2)

    print("="*70)
    print(f"Compact vocabulary saved to: {output_file}")
    print(f"Total unique entities: {vocab['total_unique_entities']}")
    print("="*70)


if __name__ == '__main__':
    main()
