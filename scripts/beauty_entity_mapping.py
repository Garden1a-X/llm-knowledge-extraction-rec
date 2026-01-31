#!/usr/bin/env python3
"""
Entity mapping functions for Beauty dataset.

Maps free-form extracted entities to standard vocabulary entities.
Used in Phase 2 extraction for entity normalization.
"""

import re


def normalize_entity(entity: str) -> str:
    """Basic normalization."""
    entity = entity.lower().strip()
    entity = re.sub(r'\s+', '_', entity)
    entity = re.sub(r'_+', '_', entity)
    entity = entity.strip('_')
    return entity


def map_to_standard_entity(entity: str, relation: str) -> str:
    """
    Map extracted entity to standard vocabulary entity.

    Args:
        entity: Raw extracted entity string
        relation: The relation type

    Returns:
        Mapped standard entity or 'other' variant
    """
    entity = normalize_entity(entity)

    if relation == 'product_type':
        return map_product_type(entity)
    elif relation == 'has_color':
        return map_color(entity)
    elif relation == 'finish_type':
        return map_finish(entity)
    elif relation == 'texture':
        return map_texture(entity)
    elif relation == 'packaging':
        return map_packaging(entity)
    elif relation == 'brand_aesthetic':
        return map_brand_aesthetic(entity)
    elif relation == 'occasion':
        return map_occasion(entity)
    elif relation == 'visual_theme':
        return map_visual_theme(entity)
    elif relation == 'additional_property':
        return map_additional_property(entity)
    else:
        return entity


def map_product_type(entity: str) -> str:
    """Map to ~15 product types."""

    # Skincare
    if re.search(r'skincare|face_mask|facial_mask|moisturizer|cleanser|serum|toner|sunscreen|'
                 r'eye_cream|face_cream|facial|anti.*aging|wrinkle|acne|skin.*care|lotion.*face|'
                 r'essence|ampoule|facial_roller|microneedling|facial_cleansing', entity):
        return 'skincare'

    # Lip products
    elif re.search(r'lip|lipstick|lip_gloss|lip_balm|lipcolor|lipbalm', entity):
        return 'lip_product'

    # Eye makeup
    elif re.search(r'eye.*shadow|eyeshadow|mascara|eyeliner|eyelash|false_eyelash|'
                   r'eye.*makeup|eye.*liner|lash|brow|eyebrow', entity):
        return 'eye_makeup'

    # Nail products
    elif re.search(r'nail|manicure|pedicure|cuticle', entity):
        return 'nail_product'

    # Haircare (washing/treatment)
    elif re.search(r'shampoo|conditioner|hair.*mask|hair.*treatment|hair.*oil|'
                   r'hair.*serum|scalp|leave.*in|detangl', entity):
        return 'haircare'

    # Hair styling
    elif re.search(r'hairspray|hair.*spray|mousse|hair.*gel|styling|hair.*wax|'
                   r'hair.*color|hair.*dye|hair.*chalk', entity):
        return 'hair_styling'

    # Hair accessories
    elif re.search(r'wig|hair.*extension|hair.*clip|headband|hair.*accessory|'
                   r'hair.*pin|barrette|scrunchie|tiara|hairband|turban|'
                   r'hair.*piece|weaving', entity):
        return 'hair_accessory'

    # Fragrance
    elif re.search(r'perfume|fragrance|cologne|essential.*oil|aromatherapy|'
                   r'scent|eau.*de|body.*mist|fragrance.*oil', entity):
        return 'fragrance'

    # Body care
    elif re.search(r'body.*lotion|body.*wash|body.*cream|deodorant|anti.*perspirant|'
                   r'body.*oil|body.*butter|hand.*cream|foot.*cream|body.*care|'
                   r'self.*tanner|tan.*accelerator|body.*scrub|exfoliat', entity):
        return 'body_care'

    # Makeup base
    elif re.search(r'foundation|concealer|powder|primer|bb.*cream|cc.*cream|'
                   r'cushion|bronzer|contour|highlight|setting|base|blush', entity):
        return 'makeup_base'

    # Jewelry
    elif re.search(r'earring|necklace|bracelet|ring|jewelry|jewellery|pendant|'
                   r'anklet|body.*piercing|belly.*ring|charm|bead', entity):
        return 'jewelry'

    # Makeup tools
    elif re.search(r'makeup.*brush|makeup.*sponge|makeup.*bag|cosmetic.*bag|'
                   r'makeup.*case|brush.*set|blender|applicator|puff|'
                   r'makeup.*mirror|compact.*mirror|makeup.*organizer|'
                   r'makeup.*pouch|cosmetic.*case', entity):
        return 'makeup_tool'

    # Hair tools
    elif re.search(r'hair.*dryer|straightener|curling.*iron|flat.*iron|'
                   r'hair.*brush|comb|hair.*clip|hairbrush|hair.*cutting|'
                   r'trimmer|clipper|shear|hair.*tool', entity):
        return 'hair_tool'

    # Bath products
    elif re.search(r'bath.*bomb|bath.*salt|soap|bar.*soap|shower.*gel|'
                   r'bubble.*bath|bath.*oil|loofah|bath.*brush|sponge|'
                   r'body.*wash.*sponge', entity):
        return 'bath_product'

    else:
        return 'product_other'


def map_color(entity: str) -> str:
    """Map to ~15 color categories."""

    if re.search(r'^black$|jet.*black|ebony|onyx|dark.*black', entity):
        return 'black'
    elif re.search(r'^white$|ivory|cream.*white|pearl.*white|snow', entity):
        return 'white'
    elif re.search(r'pink|rose|coral.*pink|blush.*pink|fuchsia|magenta|hot.*pink', entity):
        return 'pink'
    elif re.search(r'^red$|burgundy|wine|crimson|scarlet|ruby|cherry|maroon', entity):
        return 'red'
    elif re.search(r'gold|golden|champagne|bronze|copper.*gold', entity):
        return 'gold'
    elif re.search(r'silver|platinum|chrome|metallic.*gray|pewter', entity):
        return 'silver'
    elif re.search(r'brown|chocolate|tan|caramel|coffee|mocha|chestnut|amber', entity):
        return 'brown'
    elif re.search(r'nude|beige|skin.*tone|natural.*tone|peach|apricot', entity):
        return 'nude'
    elif re.search(r'blue|navy|cobalt|teal|turquoise|aqua|cyan|sapphire', entity):
        return 'blue'
    elif re.search(r'green|olive|emerald|mint|sage|forest|lime|jade', entity):
        return 'green'
    elif re.search(r'purple|violet|lavender|plum|lilac|mauve|grape|amethyst', entity):
        return 'purple'
    elif re.search(r'orange|coral(?!.*pink)|tangerine|peach.*orange|apricot.*orange|rust', entity):
        return 'orange'
    elif re.search(r'yellow|lemon|mustard|honey|butter|canary', entity):
        return 'yellow'
    elif re.search(r'multi|rainbow|varied|colorful|assorted|mixed.*color', entity):
        return 'multicolor'
    elif re.search(r'transparent|clear|translucent|see.*through|sheer', entity):
        return 'transparent'
    else:
        # Default to nearest color match
        if re.search(r'dark|deep', entity):
            return 'black'
        elif re.search(r'light|pale|pastel', entity):
            return 'white'
        else:
            return 'nude'  # neutral default


def map_finish(entity: str) -> str:
    """Map to ~8 finish types."""

    if re.search(r'matte|matt|flat|velvet.*matte|powder.*matte', entity):
        return 'matte'
    elif re.search(r'glossy|gloss|shiny|high.*shine|wet.*look|lacquer', entity):
        return 'glossy'
    elif re.search(r'shimmer|shimmery|iridescent|pearlescent|opalescent', entity):
        return 'shimmer'
    elif re.search(r'satin|silk|smooth|soft.*sheen|semi.*matte', entity):
        return 'satin'
    elif re.search(r'metallic|chrome|foil|mirror|reflective', entity):
        return 'metallic'
    elif re.search(r'glitter|sparkle|sparkly|sequin|disco|holographic', entity):
        return 'glitter'
    elif re.search(r'natural|bare|skin.*like|no.*makeup|subtle', entity):
        return 'natural'
    elif re.search(r'cream|creamy|dewy|luminous|radiant', entity):
        return 'cream'
    else:
        return 'natural'


def map_texture(entity: str) -> str:
    """Map to ~8 texture types."""

    if re.search(r'cream|creamy|rich|buttery|thick', entity):
        return 'creamy'
    elif re.search(r'liquid|fluid|watery|serum.*like|runny', entity):
        return 'liquid'
    elif re.search(r'powder|powdery|loose|pressed|dust|fine', entity):
        return 'powder'
    elif re.search(r'gel|jelly|gelatinous|clear.*gel', entity):
        return 'gel'
    elif re.search(r'solid|stick|hard|firm|wax.*like|balm', entity):
        return 'solid'
    elif re.search(r'mousse|whipped|airy|fluffy|light', entity):
        return 'mousse'
    elif re.search(r'foam|foamy|lather|bubbl', entity):
        return 'foam'
    elif re.search(r'spray|mist|aerosol', entity):
        return 'spray'
    else:
        return 'creamy'


def map_packaging(entity: str) -> str:
    """Map to ~12 packaging types."""

    if re.search(r'^tube$|squeeze.*tube|cream.*tube|metal.*tube', entity):
        return 'tube'
    elif re.search(r'jar|pot|tub|container.*jar', entity):
        return 'jar'
    elif re.search(r'pump|pump.*bottle|dispenser.*pump', entity):
        return 'pump_bottle'
    elif re.search(r'dropper|pipette|serum.*bottle|oil.*bottle', entity):
        return 'dropper_bottle'
    elif re.search(r'spray|mist.*bottle|atomizer|spritz', entity):
        return 'spray_bottle'
    elif re.search(r'compact|pressed.*powder|mirror.*case', entity):
        return 'compact'
    elif re.search(r'stick|bullet|twist.*up|roll.*on|swivel', entity):
        return 'stick'
    elif re.search(r'palette|pan|multi.*shade|eyeshadow.*set', entity):
        return 'palette'
    elif re.search(r'^box$|carton|package.*box|gift.*box', entity):
        return 'box'
    elif re.search(r'pouch|bag|sachet|packet', entity):
        return 'pouch'
    elif re.search(r'bottle|flask|vial', entity):
        return 'bottle'
    else:
        return 'container'


def map_brand_aesthetic(entity: str) -> str:
    """Map to ~8 brand aesthetic types."""

    if re.search(r'high.*end|premium|prestige|designer|exclusive|upscale', entity):
        return 'high_end'
    elif re.search(r'drugstore|affordable|budget|mass.*market|everyday', entity):
        return 'drugstore'
    elif re.search(r'natural|organic|eco|green|clean|vegan|sustainable|botanical', entity):
        return 'natural_organic'
    elif re.search(r'trendy|modern|fashion|contemporary|edgy|hip', entity):
        return 'trendy'
    elif re.search(r'professional|salon|clinical|medical|dermat', entity):
        return 'professional'
    elif re.search(r'playful|fun|cute|whimsical|colorful.*brand|youthful.*brand', entity):
        return 'playful'
    elif re.search(r'minimalist|simple|clean.*design|understated|subtle', entity):
        return 'minimalist'
    elif re.search(r'luxury|luxurious|opulent|elegant.*brand|sophisticated.*brand', entity):
        return 'luxury'
    else:
        return 'drugstore'


def map_occasion(entity: str) -> str:
    """Map to ~8 occasion types."""

    if re.search(r'everyday|daily|casual|routine|regular|work.*day', entity):
        return 'everyday'
    elif re.search(r'party|club|night.*out|celebration|festive|holiday', entity):
        return 'party'
    elif re.search(r'professional|office|work|business|corporate|formal.*work', entity):
        return 'professional'
    elif re.search(r'special.*occasion|event|gala|formal.*event|prom', entity):
        return 'special_occasion'
    elif re.search(r'wedding|bridal|bride|bridesmaid|ceremony', entity):
        return 'wedding'
    elif re.search(r'travel|vacation|trip|on.*the.*go|portable|mini', entity):
        return 'travel'
    elif re.search(r'night|evening|dinner|date.*night|glamour', entity):
        return 'night_out'
    elif re.search(r'day|daytime|morning|afternoon|brunch|lunch', entity):
        return 'daytime'
    else:
        return 'everyday'


def map_visual_theme(entity: str) -> str:
    """Map to ~8 visual theme types."""

    if re.search(r'natural.*beauty|fresh|bare|no.*makeup|minimal.*makeup|effortless', entity):
        return 'natural_beauty'
    elif re.search(r'sophisticated|refined|polished|chic|classy|timeless', entity):
        return 'sophisticated'
    elif re.search(r'youthful|young|vibrant|energetic|fresh.*faced|dewy', entity):
        return 'youthful'
    elif re.search(r'artistic|creative|avant.*garde|experimental|unique|editorial', entity):
        return 'artistic'
    elif re.search(r'bold|dramatic|statement|intense|striking|vivid', entity):
        return 'bold'
    elif re.search(r'elegant|graceful|delicate|feminine|soft.*elegant', entity):
        return 'elegant'
    elif re.search(r'cute|kawaii|sweet|adorable|pretty|girly', entity):
        return 'cute'
    elif re.search(r'glamorous|glam|hollywood|red.*carpet|luxe|diva', entity):
        return 'glamorous'
    else:
        return 'natural_beauty'


def map_additional_property(entity: str) -> str:
    """Map to ~8 additional property types."""

    if re.search(r'set|collection|kit|bundle|combo|multi.*pack', entity):
        return 'set_collection'
    elif re.search(r'travel|mini|sample|trial|deluxe.*sample|portable', entity):
        return 'travel_size'
    elif re.search(r'full.*size|regular|standard', entity):
        return 'full_size'
    elif re.search(r'sample|tester|trial.*size', entity):
        return 'sample'
    elif re.search(r'refill|replacement|cartridge', entity):
        return 'refill'
    elif re.search(r'limited.*edition|exclusive|seasonal|collaboration', entity):
        return 'limited_edition'
    elif re.search(r'gift|present|holiday.*set', entity):
        return 'gift_set'
    else:
        return 'property_other'


# Export mapping functions for use in extraction script
__all__ = [
    'normalize_entity',
    'map_to_standard_entity',
    'map_product_type',
    'map_color',
    'map_finish',
    'map_texture',
    'map_packaging',
    'map_brand_aesthetic',
    'map_occasion',
    'map_visual_theme',
    'map_additional_property'
]
