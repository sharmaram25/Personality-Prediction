ARCHETYPES = {
    "The Architect": {
        "description": "Imaginative and strategic thinkers, with a plan for everything. They are analytical, creative, and logical.",
        "rules": {"OPN": "high", "CSN": "high", "EXT": "low", "AGR": "low", "EST": "low"}
    },
    "The Pathfinder": {
        "description": "Charismatic and inspiring leaders, able to mesmerize their listeners. They are driven, energetic, and sociable.",
        "rules": {"EXT": "high", "OPN": "high", "AGR": "high", "CSN": "any", "EST": "low"}
    },
    "The Mediator": {
        "description": "Poetic, kind, and altruistic people, always eager to help a good cause. They are empathetic, caring, and idealistic.",
        "rules": {"AGR": "high", "OPN": "high", "EXT": "any", "CSN": "any", "EST": "any"}
    },
    "The Sentinel": {
        "description": "Practical and fact-minded individuals, whose reliability cannot be doubted. They are organized, responsible, and traditional.",
        "rules": {"CSN": "high", "AGR": "high", "EXT": "any", "OPN": "low", "EST": "low"}
    },
    "The Virtuoso": {
        "description": "Bold and practical experimenters, masters of all kinds of tools. They are hands-on, inventive, and independent.",
        "rules": {"OPN": "high", "EXT": "low", "CSN": "any", "AGR": "low", "EST": "any"}
    },
    "The Adventurer": {
        "description": "Flexible and charming artists, always ready to explore and experience something new. They are spontaneous, energetic, and enthusiastic.",
        "rules": {"EXT": "high", "OPN": "high", "EST": "low", "AGR": "any", "CSN": "low"}
    },
    "The Logician": {
        "description": "Innovative inventors with an unquenchable thirst for knowledge. They are precise, abstract thinkers who enjoy deep analysis.",
        "rules": {"OPN": "high", "CSN": "high", "EXT": "low", "AGR": "low", "EST": "any"}
    },
    "The Commander": {
        "description": "Bold, imaginative and strong-willed leaders, always finding a way – or making one. They are decisive, confident, and strategic.",
        "rules": {"EXT": "high", "CSN": "high", "OPN": "any", "AGR": "low", "EST": "low"}
    },
    "The Advocate": {
        "description": "Quiet and mystical, yet very inspiring and tireless idealists. They are compassionate, insightful, and dedicated.",
        "rules": {"OPN": "high", "AGR": "high", "EXT": "low", "CSN": "any", "EST": "any"}
    },
    "The Entertainer": {
        "description": "Spontaneous, energetic, and enthusiastic people – life is never boring around them. They are outgoing, fun-loving, and generous.",
        "rules": {"EXT": "high", "AGR": "high", "CSN": "low", "OPN": "any", "EST": "any"}
    },
    "The Realist": {
        "description": "Practical and down-to-earth, they are reliable and steady individuals who prefer facts to fiction.",
        "rules": {"CSN": "high", "OPN": "low", "EXT": "any", "AGR": "any", "EST": "low"}
    },
    "The Visionary": {
        "description": "Creative and insightful, they see possibilities everywhere. They are driven by their values and seek to inspire others.",
        "rules": {"OPN": "high", "AGR": "high", "EXT": "high", "CSN": "any", "EST": "any"}
    }
}

def get_archetype(scores, threshold=3.0):
    """
    Maps OCEAN scores to a personality archetype based on high/low trait thresholds.

    Args:
        scores (dict): A dictionary with OCEAN traits as keys and scores as values.
        threshold (float): The value to determine if a trait score is high or low.

    Returns:
        tuple: A tuple containing the archetype name and its description.
    """
    trait_levels = {
        trait: "high" if score >= threshold else "low" for trait, score in scores.items()
    }

    for name, archetype in ARCHETYPES.items():
        match = True
        for trait, level in archetype["rules"].items():
            if level != "any" and trait_levels.get(trait) != level:
                match = False
                break
        if match:
            return name, archetype["description"]
    
    return "The Unique", "Your combination of traits is unique and doesn't fit a predefined archetype."

if __name__ == '__main__':
    # Example Usage
    sample_scores = {'EXT': 4.5, 'EST': 2.1, 'AGR': 3.8, 'CSN': 4.2, 'OPN': 3.9}
    archetype_name, archetype_desc = get_archetype(sample_scores)
    print(f"Archetype: {archetype_name}")
    print(f"Description: {archetype_desc}")

    sample_scores_2 = {'EXT': 2.0, 'EST': 4.0, 'AGR': 2.5, 'CSN': 2.8, 'OPN': 4.5}
    archetype_name, archetype_desc = get_archetype(sample_scores_2)
    print(f"\nArchetype: {archetype_name}")
    print(f"Description: {archetype_desc}")
