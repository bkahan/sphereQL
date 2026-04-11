"""
Built-in dataset for the SphereQL quickstart example.

100 sentences across 10 topics with deterministic 64-d embeddings.
No external dependencies — embeddings are generated via FNV-1a hashing.
"""

STOPWORDS = {
    "the", "and", "with", "for", "its", "this", "that", "from", "are",
    "was", "were", "has", "have", "had", "but", "not", "can", "all",
    "will", "one", "two", "their", "which", "when", "been", "would",
    "each", "make", "than", "into", "over", "such", "after", "also",
}


def encode(text, dim=64):
    """Deterministic bag-of-words encoder via FNV-1a hashing."""
    bag = [0.0] * dim
    for token in text.lower().split():
        token = "".join(c for c in token if c.isalpha())
        if len(token) < 3 or token in STOPWORDS:
            continue
        h = 0xCBF29CE484222325
        for b in token.encode():
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        bag[h % dim] += 1.0
    mag = sum(x * x for x in bag) ** 0.5
    if mag > 0:
        bag = [x / mag for x in bag]
    return bag


SENTENCES = [
    # ── Science ──────────────────────────────────────────────────────
    {"text": "The double-slit experiment demonstrates that light exhibits both wave and particle properties.", "category": "science"},
    {"text": "CRISPR-Cas9 enables precise editing of DNA sequences by using a guide RNA to target specific genomic locations.", "category": "science"},
    {"text": "Black holes form when massive stars collapse under their own gravity at the end of their nuclear fuel cycle.", "category": "science"},
    {"text": "Quantum entanglement allows two particles to share correlated states regardless of the distance between them.", "category": "science"},
    {"text": "The periodic table organizes elements by atomic number and electron configuration into groups with similar properties.", "category": "science"},
    {"text": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from sunlight.", "category": "science"},
    {"text": "General relativity predicts that massive objects curve spacetime, causing nearby objects to follow geodesic paths.", "category": "science"},
    {"text": "Mitochondria generate adenosine triphosphate through oxidative phosphorylation in eukaryotic cells.", "category": "science"},
    {"text": "The Heisenberg uncertainty principle states that position and momentum cannot both be precisely measured simultaneously.", "category": "science"},
    {"text": "Plate tectonics explains how continental drift, earthquakes, and volcanic activity arise from moving lithospheric plates.", "category": "science"},

    # ── Technology ───────────────────────────────────────────────────
    {"text": "Transformer architectures use self-attention mechanisms to process sequences in parallel rather than sequentially.", "category": "technology"},
    {"text": "Kubernetes orchestrates containerized applications across clusters by managing deployment, scaling, and networking.", "category": "technology"},
    {"text": "Solid-state drives store data in NAND flash memory cells, offering faster access times than magnetic hard drives.", "category": "technology"},
    {"text": "TCP guarantees reliable data delivery by using sequence numbers, acknowledgments, and retransmission of lost packets.", "category": "technology"},
    {"text": "Graphics processing units accelerate parallel computations using thousands of small cores optimized for matrix operations.", "category": "technology"},
    {"text": "Public key cryptography enables secure communication by using asymmetric key pairs for encryption and digital signatures.", "category": "technology"},
    {"text": "WebAssembly provides a portable binary format that allows near-native execution speed in web browsers.", "category": "technology"},
    {"text": "Distributed hash tables partition key-value data across multiple nodes to achieve horizontal scalability.", "category": "technology"},
    {"text": "Garbage collectors automatically reclaim memory by tracing reachable objects and freeing unreferenced allocations.", "category": "technology"},
    {"text": "Convolutional neural networks apply learned filters across spatial dimensions to detect patterns in image data.", "category": "technology"},

    # ── Cooking ──────────────────────────────────────────────────────
    {"text": "Caramelizing onions over low heat for forty minutes develops deep, sweet flavors that transform a simple soup.", "category": "cooking"},
    {"text": "Folding egg whites gently into batter preserves air bubbles that give souffles their characteristic rise.", "category": "cooking"},
    {"text": "Brining a turkey in salt water for twelve hours ensures the meat stays moist during roasting.", "category": "cooking"},
    {"text": "Deglazing a pan with wine after searing dissolves the fond and creates a rich, flavorful sauce base.", "category": "cooking"},
    {"text": "Tempering chocolate requires carefully cycling between heating and cooling to form stable cocoa butter crystals.", "category": "cooking"},
    {"text": "Fermented sourdough starter relies on wild yeast and lactobacilli to leaven bread without commercial yeast.", "category": "cooking"},
    {"text": "Blanching vegetables in boiling water followed by an ice bath preserves their color and stops enzymatic browning.", "category": "cooking"},
    {"text": "Emulsifying oil and vinegar with mustard creates a stable vinaigrette that resists separation.", "category": "cooking"},
    {"text": "Braising tough cuts of meat at low temperature breaks down collagen into gelatin, producing tender results.", "category": "cooking"},
    {"text": "Toasting spices in a dry skillet before grinding releases volatile aromatic compounds that intensify flavor.", "category": "cooking"},

    # ── Sports ───────────────────────────────────────────────────────
    {"text": "Zone defense in basketball assigns each player a specific area of the court to guard rather than an individual opponent.", "category": "sports"},
    {"text": "The offside rule in soccer prevents attacking players from positioning behind the last defender before the ball is played.", "category": "sports"},
    {"text": "A decathlon combines ten track and field events to measure overall athletic ability across sprinting, jumping, and throwing.", "category": "sports"},
    {"text": "Periodization structures athletic training into cycles of increasing intensity followed by recovery phases.", "category": "sports"},
    {"text": "The Duckworth-Lewis method recalculates target scores in rain-interrupted cricket matches using statistical models.", "category": "sports"},
    {"text": "Plyometric exercises like box jumps train the stretch-shortening cycle of muscles to increase explosive power.", "category": "sports"},
    {"text": "The Elo rating system quantifies relative skill levels in chess by updating scores after each competitive match.", "category": "sports"},
    {"text": "Wind sprints alternate between maximum effort and brief recovery periods to build anaerobic capacity.", "category": "sports"},
    {"text": "A scrum in rugby involves eight players from each team binding together to contest possession of the ball.", "category": "sports"},
    {"text": "Drafting in cycling reduces air resistance by riding closely behind another cyclist at high speeds.", "category": "sports"},

    # ── Music ────────────────────────────────────────────────────────
    {"text": "A fugue develops a short melodic subject through systematic imitation across multiple independent voices.", "category": "music"},
    {"text": "The twelve-bar blues progression follows a specific pattern of tonic, subdominant, and dominant chords.", "category": "music"},
    {"text": "Equal temperament tuning divides the octave into twelve equal semitones, enabling modulation between any key.", "category": "music"},
    {"text": "Syncopation creates rhythmic interest by emphasizing beats or subdivisions that are normally unaccented.", "category": "music"},
    {"text": "Orchestral scoring assigns melodic lines and harmonies to different instrument families based on timbre and range.", "category": "music"},
    {"text": "Polyrhythm layers two or more contrasting rhythmic patterns simultaneously, common in West African drumming traditions.", "category": "music"},
    {"text": "The sonata form structures a movement into exposition, development, and recapitulation sections.", "category": "music"},
    {"text": "Overtones above the fundamental frequency determine the distinctive timbre of each musical instrument.", "category": "music"},
    {"text": "Modal jazz improvisation explores scales like Dorian and Mixolydian rather than following rapid chord changes.", "category": "music"},
    {"text": "A crescendo gradually increases the volume of a passage, building tension and emotional intensity.", "category": "music"},

    # ── History ──────────────────────────────────────────────────────
    {"text": "The Treaty of Westphalia in 1648 established the principle of state sovereignty in international relations.", "category": "history"},
    {"text": "The Silk Road connected China to the Mediterranean, facilitating trade in goods, ideas, and cultural practices.", "category": "history"},
    {"text": "The French Revolution abolished the feudal system and established principles of citizenship and popular sovereignty.", "category": "history"},
    {"text": "Gutenberg's printing press in 1440 enabled mass production of books and accelerated the spread of literacy.", "category": "history"},
    {"text": "The Congress of Vienna in 1815 redrew European borders after the Napoleonic Wars to restore balance of power.", "category": "history"},
    {"text": "The Rosetta Stone provided the key to deciphering Egyptian hieroglyphics through its parallel Greek inscription.", "category": "history"},
    {"text": "The Industrial Revolution transformed manufacturing from cottage industries to mechanized factory production.", "category": "history"},
    {"text": "The Magna Carta of 1215 limited royal authority and established that even the king was subject to law.", "category": "history"},
    {"text": "The Columbian Exchange transferred crops, livestock, and diseases between the Old World and the Americas.", "category": "history"},
    {"text": "The Byzantine Empire preserved Roman legal traditions and Greek scholarship throughout the medieval period.", "category": "history"},

    # ── Nature ───────────────────────────────────────────────────────
    {"text": "Coral reefs support roughly a quarter of all marine species despite covering less than one percent of the ocean floor.", "category": "nature"},
    {"text": "Monarch butterflies migrate up to three thousand miles from Canada to central Mexico each autumn.", "category": "nature"},
    {"text": "Mycorrhizal fungi form symbiotic networks with tree roots, exchanging minerals for photosynthetic sugars.", "category": "nature"},
    {"text": "The Amazon rainforest produces approximately twenty percent of the world's oxygen through photosynthetic activity.", "category": "nature"},
    {"text": "Wolves reintroduced to Yellowstone triggered a trophic cascade that restored riverbank vegetation and stabilized erosion.", "category": "nature"},
    {"text": "Deep-sea hydrothermal vents support ecosystems based on chemosynthesis rather than sunlight-driven photosynthesis.", "category": "nature"},
    {"text": "The circadian rhythm synchronizes biological processes to the twenty-four-hour light-dark cycle using internal clocks.", "category": "nature"},
    {"text": "Mangrove forests protect coastlines from storm surges while serving as nursery habitats for juvenile fish.", "category": "nature"},
    {"text": "Bioluminescent organisms produce light through chemical reactions involving luciferin and the enzyme luciferase.", "category": "nature"},
    {"text": "Old-growth forests sequester significantly more carbon per hectare than younger plantation forests.", "category": "nature"},

    # ── Health ───────────────────────────────────────────────────────
    {"text": "Vaccines train the adaptive immune system to recognize pathogens by presenting weakened or inactivated antigens.", "category": "health"},
    {"text": "Regular aerobic exercise strengthens cardiac muscle and improves the efficiency of oxygen delivery to tissues.", "category": "health"},
    {"text": "The gut microbiome influences digestion, immune function, and even mood through the gut-brain axis.", "category": "health"},
    {"text": "Sleep deprivation impairs cognitive function, weakens immune response, and increases the risk of chronic disease.", "category": "health"},
    {"text": "Omega-three fatty acids reduce systemic inflammation and support cardiovascular health when consumed regularly.", "category": "health"},
    {"text": "Resistance training increases bone mineral density, reducing the risk of osteoporosis in older adults.", "category": "health"},
    {"text": "Chronic stress elevates cortisol levels, which can suppress immune function and accelerate cellular aging.", "category": "health"},
    {"text": "Antioxidants neutralize free radicals that cause oxidative damage to cellular membranes and DNA.", "category": "health"},
    {"text": "Mindfulness meditation has been shown to reduce anxiety and improve attention regulation in clinical trials.", "category": "health"},
    {"text": "Adequate hydration supports kidney function, thermoregulation, and nutrient transport throughout the body.", "category": "health"},

    # ── Philosophy ───────────────────────────────────────────────────
    {"text": "Descartes' cogito ergo sum established thinking as the foundational certainty from which knowledge can be built.", "category": "philosophy"},
    {"text": "Utilitarianism judges the morality of an action by whether it maximizes overall happiness or well-being.", "category": "philosophy"},
    {"text": "Kant's categorical imperative demands that moral rules must be universalizable without logical contradiction.", "category": "philosophy"},
    {"text": "Existentialism holds that existence precedes essence, meaning individuals define their own purpose through choices.", "category": "philosophy"},
    {"text": "The trolley problem illustrates the tension between consequentialist and deontological approaches to ethical dilemmas.", "category": "philosophy"},
    {"text": "Socratic questioning uses systematic inquiry to expose assumptions and arrive at deeper understanding.", "category": "philosophy"},
    {"text": "Pragmatism evaluates the truth of beliefs by their practical consequences and usefulness in lived experience.", "category": "philosophy"},
    {"text": "The ship of Theseus asks whether an object that has had all its components replaced remains the same object.", "category": "philosophy"},
    {"text": "Phenomenology studies the structures of consciousness and the ways objects appear to subjective experience.", "category": "philosophy"},
    {"text": "The veil of ignorance thought experiment asks people to design fair institutions without knowing their position in society.", "category": "philosophy"},

    # ── Business ─────────────────────────────────────────────────────
    {"text": "Compound interest allows investment returns to generate their own returns, creating exponential growth over time.", "category": "business"},
    {"text": "Supply chain diversification reduces the risk of disruption by sourcing materials from multiple geographic regions.", "category": "business"},
    {"text": "Network effects increase the value of a product as more users adopt it, creating winner-take-all market dynamics.", "category": "business"},
    {"text": "Marginal cost pricing sets product prices based on the cost of producing one additional unit rather than average costs.", "category": "business"},
    {"text": "Double-entry bookkeeping records every financial transaction as equal debits and credits to maintain balanced accounts.", "category": "business"},
    {"text": "Venture capital firms invest in early-stage companies in exchange for equity stakes with high growth potential.", "category": "business"},
    {"text": "Opportunity cost measures the value of the next best alternative foregone when making an economic decision.", "category": "business"},
    {"text": "Agile project management uses iterative sprints and continuous feedback to adapt to changing requirements.", "category": "business"},
    {"text": "Market segmentation divides consumers into distinct groups based on demographics, behavior, or needs.", "category": "business"},
    {"text": "Economies of scale reduce per-unit costs as production volume increases, giving larger firms a competitive advantage.", "category": "business"},
]

# Pre-compute embeddings
for item in SENTENCES:
    item["embedding"] = encode(item["text"])
