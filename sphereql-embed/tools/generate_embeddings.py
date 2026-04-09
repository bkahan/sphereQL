#!/usr/bin/env python3
"""Generate sentence embeddings using all-MiniLM-L6-v2 for the sphereql-embed demo.

Usage:
    python3 generate_embeddings.py                       # writes to stdout
    python3 generate_embeddings.py > embeddings.json     # save to file
"""

import json
import sys
from sentence_transformers import SentenceTransformer

SENTENCES = [
    # --- Science (8) ---
    {"id": "sci-1", "category": "science", "text": "The speed of light in a vacuum is approximately 299,792 kilometers per second."},
    {"id": "sci-2", "category": "science", "text": "DNA contains the genetic instructions for the development of all known living organisms."},
    {"id": "sci-3", "category": "science", "text": "Quantum entanglement allows particles to be correlated instantaneously across vast distances."},
    {"id": "sci-4", "category": "science", "text": "Einstein's theory of general relativity describes gravity as the curvature of spacetime."},
    {"id": "sci-5", "category": "science", "text": "The human brain contains approximately 86 billion neurons connected by trillions of synapses."},
    {"id": "sci-6", "category": "science", "text": "Photosynthesis converts sunlight into chemical energy stored in glucose molecules."},
    {"id": "sci-7", "category": "science", "text": "Black holes are regions of spacetime where gravity is so strong that nothing can escape."},
    {"id": "sci-8", "category": "science", "text": "The periodic table organizes chemical elements by their atomic number and electron configuration."},

    # --- Technology (8) ---
    {"id": "tech-1", "category": "technology", "text": "Machine learning algorithms improve their performance by training on large datasets."},
    {"id": "tech-2", "category": "technology", "text": "The internet connects billions of devices through a global network of computer systems."},
    {"id": "tech-3", "category": "technology", "text": "Artificial intelligence can now generate realistic images and human-like text responses."},
    {"id": "tech-4", "category": "technology", "text": "Blockchain technology provides a decentralized and tamper-resistant digital ledger system."},
    {"id": "tech-5", "category": "technology", "text": "Autonomous vehicles use sensors and algorithms to navigate roads without human input."},
    {"id": "tech-6", "category": "technology", "text": "Cloud computing allows on-demand access to shared computing resources over the internet."},
    {"id": "tech-7", "category": "technology", "text": "Quantum computers use qubits to solve certain problems exponentially faster than classical machines."},
    {"id": "tech-8", "category": "technology", "text": "Cybersecurity protects computer systems and networks from digital attacks and data breaches."},

    # --- Sports (7) ---
    {"id": "sport-1", "category": "sports", "text": "The Olympic Games bring together athletes from around the world to compete every four years."},
    {"id": "sport-2", "category": "sports", "text": "A marathon covers 26.2 miles and tests the endurance limits of long-distance runners."},
    {"id": "sport-3", "category": "sports", "text": "Basketball requires teamwork, agility, and precise shooting to outscore the opponent."},
    {"id": "sport-4", "category": "sports", "text": "Soccer is the most popular sport in the world with over four billion fans."},
    {"id": "sport-5", "category": "sports", "text": "Tennis matches can last for hours as players rally the ball across the net."},
    {"id": "sport-6", "category": "sports", "text": "Swimming is an excellent full-body cardiovascular exercise suitable for all ages."},
    {"id": "sport-7", "category": "sports", "text": "Professional athletes dedicate years of intense training to compete at the highest level."},

    # --- Cooking (7) ---
    {"id": "cook-1", "category": "cooking", "text": "Preheat the oven to 375 degrees and line the baking sheet with parchment paper."},
    {"id": "cook-2", "category": "cooking", "text": "Fresh herbs like basil and cilantro add vibrant flavor to almost any dish."},
    {"id": "cook-3", "category": "cooking", "text": "A good risotto requires patience and constant stirring to achieve a creamy texture."},
    {"id": "cook-4", "category": "cooking", "text": "Marinating meat overnight allows the flavors to penetrate deeply into the protein."},
    {"id": "cook-5", "category": "cooking", "text": "The secret to a perfect pie crust is using very cold butter and minimal handling."},
    {"id": "cook-6", "category": "cooking", "text": "Sauteing onions slowly until caramelized brings out their natural sweetness."},
    {"id": "cook-7", "category": "cooking", "text": "Homemade pasta has a remarkably different taste and texture from store-bought varieties."},

    # --- Arts & Music (7) ---
    {"id": "art-1", "category": "arts", "text": "Beethoven composed his magnificent Ninth Symphony while almost completely deaf."},
    {"id": "art-2", "category": "arts", "text": "Jazz music originated in New Orleans blending African rhythms with European harmony."},
    {"id": "art-3", "category": "arts", "text": "The Impressionist painters captured light and atmosphere with loose, visible brushstrokes."},
    {"id": "art-4", "category": "arts", "text": "A symphony orchestra typically includes strings, woodwinds, brass, and percussion sections."},
    {"id": "art-5", "category": "arts", "text": "Shakespeare's plays have been performed continuously around the world for over four centuries."},
    {"id": "art-6", "category": "arts", "text": "Modern dance breaks away from classical ballet with expressive, free-form movement."},
    {"id": "art-7", "category": "arts", "text": "Photography transformed how humanity documents and shares visual experiences."},

    # --- Nature (7) ---
    {"id": "nat-1", "category": "nature", "text": "The Amazon rainforest produces about twenty percent of the world's oxygen supply."},
    {"id": "nat-2", "category": "nature", "text": "Coral reefs support more species per unit area than any other marine environment."},
    {"id": "nat-3", "category": "nature", "text": "Bird migration patterns span thousands of miles across multiple continents each year."},
    {"id": "nat-4", "category": "nature", "text": "Volcanic eruptions release gases and molten rock from deep within the Earth's mantle."},
    {"id": "nat-5", "category": "nature", "text": "Ocean currents play a crucial role in regulating global climate and weather patterns."},
    {"id": "nat-6", "category": "nature", "text": "Old-growth forests contain complex ecosystems that have developed over centuries."},
    {"id": "nat-7", "category": "nature", "text": "The water cycle continuously moves water between the atmosphere, land, and oceans."},

    # --- History (6) ---
    {"id": "hist-1", "category": "history", "text": "The Renaissance saw a dramatic revival of art, science, and classical learning in Europe."},
    {"id": "hist-2", "category": "history", "text": "The Industrial Revolution transformed manufacturing through steam power and mechanization."},
    {"id": "hist-3", "category": "history", "text": "Ancient Rome built an extensive network of roads and aqueducts across its vast empire."},
    {"id": "hist-4", "category": "history", "text": "The invention of the printing press revolutionized the spread of knowledge and literacy."},
    {"id": "hist-5", "category": "history", "text": "European exploration in the fifteenth century established new trade routes across the globe."},
    {"id": "hist-6", "category": "history", "text": "The French Revolution fundamentally transformed the political structure of France and Europe."},
]


QUERIES = [
    {"id": "q-gravity", "text": "How does gravity work in space?"},
    {"id": "q-recipes", "text": "What recipes should I try for dinner?"},
    {"id": "q-musicians", "text": "Tell me about famous musicians and composers"},
]


def main():
    print(f"Loading model all-MiniLM-L6-v2...", file=sys.stderr)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [s["text"] for s in SENTENCES]
    query_texts = [q["text"] for q in QUERIES]
    print(f"Encoding {len(texts)} sentences + {len(query_texts)} queries...", file=sys.stderr)

    all_texts = texts + query_texts
    all_embeddings = model.encode(all_texts, show_progress_bar=False)
    embeddings = all_embeddings[: len(texts)]
    query_embeddings = all_embeddings[len(texts) :]

    output = {
        "model": "all-MiniLM-L6-v2",
        "dimension": int(embeddings.shape[1]),
        "sentences": [],
        "queries": [],
    }

    for sent, emb in zip(SENTENCES, embeddings):
        output["sentences"].append({
            "id": sent["id"],
            "text": sent["text"],
            "category": sent["category"],
            "embedding": [round(float(x), 6) for x in emb],
        })

    for q, emb in zip(QUERIES, query_embeddings):
        output["queries"].append({
            "id": q["id"],
            "text": q["text"],
            "embedding": [round(float(x), 6) for x in emb],
        })

    json.dump(output, sys.stdout, indent=None, separators=(",", ":"))
    print(f"\nDone: {len(texts)} sentences + {len(query_texts)} queries, {embeddings.shape[1]}-d", file=sys.stderr)


if __name__ == "__main__":
    main()
