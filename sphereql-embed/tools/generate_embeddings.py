#!/usr/bin/env python3
"""Generate 200 sentence embeddings using all-MiniLM-L6-v2 for the sphereql-embed demo.

Usage:
    python3 generate_embeddings.py                       # writes to stdout
    python3 generate_embeddings.py > embeddings.json     # save to file
"""

import json
import sys
from sentence_transformers import SentenceTransformer

SENTENCES = [
    # ── Science (20) ─────────────────────────────────────────────────────
    {"id":"sci-01","category":"science","text":"The speed of light in a vacuum is approximately 299,792 kilometers per second."},
    {"id":"sci-02","category":"science","text":"DNA carries the genetic instructions for the development of all known living organisms."},
    {"id":"sci-03","category":"science","text":"Quantum entanglement allows particles to be correlated instantaneously across vast distances."},
    {"id":"sci-04","category":"science","text":"Einstein's theory of general relativity describes gravity as the curvature of spacetime."},
    {"id":"sci-05","category":"science","text":"The human brain contains approximately 86 billion neurons connected by trillions of synapses."},
    {"id":"sci-06","category":"science","text":"Photosynthesis converts sunlight into chemical energy stored in glucose molecules."},
    {"id":"sci-07","category":"science","text":"Black holes are regions of spacetime where gravity is so strong that nothing can escape."},
    {"id":"sci-08","category":"science","text":"The periodic table organizes chemical elements by their atomic number and electron configuration."},
    {"id":"sci-09","category":"science","text":"Mitochondria generate most of the cell's supply of adenosine triphosphate through oxidative phosphorylation."},
    {"id":"sci-10","category":"science","text":"The Higgs boson gives other fundamental particles their mass through interaction with the Higgs field."},
    {"id":"sci-11","category":"science","text":"Plate tectonics explains how the Earth's lithosphere is divided into large plates that slowly move over the asthenosphere."},
    {"id":"sci-12","category":"science","text":"Entropy in a closed system tends to increase over time according to the second law of thermodynamics."},
    {"id":"sci-13","category":"science","text":"The double-helix structure of DNA was first described by Watson and Crick in 1953."},
    {"id":"sci-14","category":"science","text":"Neutrinos are nearly massless subatomic particles that rarely interact with ordinary matter."},
    {"id":"sci-15","category":"science","text":"How does the theory of evolution explain the diversity of life on Earth?"},
    {"id":"sci-16","category":"science","text":"What causes the northern lights to appear in polar regions of the sky?"},
    {"id":"sci-17","category":"science","text":"Why does the uncertainty principle place a fundamental limit on measuring quantum systems?"},
    {"id":"sci-18","category":"science","text":"How do vaccines train the immune system to recognize and fight specific pathogens?"},
    {"id":"sci-19","category":"science","text":"What evidence supports the theory that the universe originated from the Big Bang?"},
    {"id":"sci-20","category":"science","text":"Can stem cells be reliably programmed to regenerate damaged tissues in the human body?"},

    # ── Technology (20) ──────────────────────────────────────────────────
    {"id":"tech-01","category":"technology","text":"Machine learning algorithms improve their performance by training on large datasets."},
    {"id":"tech-02","category":"technology","text":"The internet connects billions of devices through a global network of computer systems."},
    {"id":"tech-03","category":"technology","text":"Artificial intelligence can now generate realistic images and human-like text responses."},
    {"id":"tech-04","category":"technology","text":"Blockchain technology provides a decentralized and tamper-resistant digital ledger system."},
    {"id":"tech-05","category":"technology","text":"Autonomous vehicles use sensors and neural networks to navigate roads without human input."},
    {"id":"tech-06","category":"technology","text":"Cloud computing allows on-demand access to shared computing resources over the internet."},
    {"id":"tech-07","category":"technology","text":"Quantum computers use qubits to solve certain problems exponentially faster than classical machines."},
    {"id":"tech-08","category":"technology","text":"Cybersecurity protects computer systems and networks from digital attacks and data breaches."},
    {"id":"tech-09","category":"technology","text":"Natural language processing enables computers to understand and generate human language."},
    {"id":"tech-10","category":"technology","text":"The global positioning system uses a constellation of satellites to determine precise locations on Earth."},
    {"id":"tech-11","category":"technology","text":"Three-dimensional printing builds physical objects layer by layer from digital blueprints."},
    {"id":"tech-12","category":"technology","text":"Edge computing processes data closer to where it is generated rather than in a centralized data center."},
    {"id":"tech-13","category":"technology","text":"Reinforcement learning agents discover optimal strategies through trial and error in simulated environments."},
    {"id":"tech-14","category":"technology","text":"The Internet of Things connects everyday appliances and industrial equipment to the web for remote monitoring."},
    {"id":"tech-15","category":"technology","text":"How do convolutional neural networks recognize objects in photographs?"},
    {"id":"tech-16","category":"technology","text":"What security risks does widespread adoption of smart home devices introduce?"},
    {"id":"tech-17","category":"technology","text":"Can artificial general intelligence ever match the flexibility of human reasoning?"},
    {"id":"tech-18","category":"technology","text":"How does end-to-end encryption protect private messages from being intercepted?"},
    {"id":"tech-19","category":"technology","text":"What role do graphics processing units play in accelerating deep learning training?"},
    {"id":"tech-20","category":"technology","text":"Why is version control essential for collaborative software development projects?"},

    # ── Sports (20) ──────────────────────────────────────────────────────
    {"id":"sport-01","category":"sports","text":"The Olympic Games bring together athletes from around the world to compete every four years."},
    {"id":"sport-02","category":"sports","text":"A marathon covers 26.2 miles and tests the endurance limits of long-distance runners."},
    {"id":"sport-03","category":"sports","text":"Basketball requires teamwork, agility, and precise shooting to outscore the opposing team."},
    {"id":"sport-04","category":"sports","text":"Soccer is the most widely followed sport in the world, with over four billion fans."},
    {"id":"sport-05","category":"sports","text":"Tennis matches can last for hours as players rally the ball across the net."},
    {"id":"sport-06","category":"sports","text":"Swimming is an excellent full-body cardiovascular exercise suitable for people of all ages."},
    {"id":"sport-07","category":"sports","text":"Professional athletes dedicate years of intense training to compete at the highest level."},
    {"id":"sport-08","category":"sports","text":"A well-designed strength training program builds muscle, strengthens joints, and prevents injuries."},
    {"id":"sport-09","category":"sports","text":"Ice hockey combines skating speed, stick handling, and physical contact in a fast-paced arena."},
    {"id":"sport-10","category":"sports","text":"Gymnastics demands extraordinary flexibility, balance, and explosive power from its competitors."},
    {"id":"sport-11","category":"sports","text":"Cricket matches can unfold over five days in the traditional test format of the game."},
    {"id":"sport-12","category":"sports","text":"Rock climbing challenges both physical strength and mental problem-solving on vertical terrain."},
    {"id":"sport-13","category":"sports","text":"Interval training alternates bursts of high-intensity effort with periods of active recovery."},
    {"id":"sport-14","category":"sports","text":"The Tour de France is the most prestigious multi-stage cycling race in professional road cycling."},
    {"id":"sport-15","category":"sports","text":"How does altitude training improve an athlete's oxygen-carrying capacity?"},
    {"id":"sport-16","category":"sports","text":"What strategies help prevent overuse injuries in endurance athletes?"},
    {"id":"sport-17","category":"sports","text":"Why is proper hydration so critical for sustaining peak athletic performance?"},
    {"id":"sport-18","category":"sports","text":"How do referees use video replay technology to make fairer officiating decisions?"},
    {"id":"sport-19","category":"sports","text":"What mental techniques do elite competitors use to stay focused under pressure?"},
    {"id":"sport-20","category":"sports","text":"Can data analytics give sports teams a measurable competitive advantage?"},

    # ── Cooking (20) ─────────────────────────────────────────────────────
    {"id":"cook-01","category":"cooking","text":"Preheat the oven to 375 degrees and line the baking sheet with parchment paper."},
    {"id":"cook-02","category":"cooking","text":"Fresh herbs like basil and cilantro add vibrant flavor to almost any savory dish."},
    {"id":"cook-03","category":"cooking","text":"A good risotto requires patience and constant stirring to achieve a creamy texture."},
    {"id":"cook-04","category":"cooking","text":"Marinating meat overnight allows the flavors to penetrate deeply into the protein."},
    {"id":"cook-05","category":"cooking","text":"The secret to a perfect pie crust is using very cold butter and minimal handling of the dough."},
    {"id":"cook-06","category":"cooking","text":"Sauteing onions slowly until caramelized brings out their natural sweetness."},
    {"id":"cook-07","category":"cooking","text":"Homemade pasta has a remarkably different taste and texture from store-bought dried varieties."},
    {"id":"cook-08","category":"cooking","text":"A cast-iron skillet retains heat evenly and develops a natural nonstick surface over time."},
    {"id":"cook-09","category":"cooking","text":"Fermented foods like kimchi and sauerkraut support healthy digestion with beneficial bacteria."},
    {"id":"cook-10","category":"cooking","text":"Reducing a stock over low heat concentrates its flavor into a rich, glossy sauce."},
    {"id":"cook-11","category":"cooking","text":"Sourdough bread relies on a live culture of wild yeast and lactic acid bacteria for leavening."},
    {"id":"cook-12","category":"cooking","text":"Blanching vegetables in boiling water and then plunging them into ice water preserves their bright color."},
    {"id":"cook-13","category":"cooking","text":"A sharp knife is safer than a dull one because it requires less force and slips less often."},
    {"id":"cook-14","category":"cooking","text":"Emulsification binds oil and vinegar together with the help of an agent like egg yolk or mustard."},
    {"id":"cook-15","category":"cooking","text":"How does the Maillard reaction create the crispy brown crust on seared steaks?"},
    {"id":"cook-16","category":"cooking","text":"What makes sourdough fermentation produce a tangier flavor than commercial yeast breads?"},
    {"id":"cook-17","category":"cooking","text":"Why does tempering chocolate require precise temperature control to achieve a glossy finish?"},
    {"id":"cook-18","category":"cooking","text":"Which spices pair best with roasted root vegetables for a hearty autumn side dish?"},
    {"id":"cook-19","category":"cooking","text":"How can home cooks safely preserve seasonal produce through canning and pickling?"},
    {"id":"cook-20","category":"cooking","text":"What techniques produce the flakiest layers in laminated pastry doughs like croissants?"},

    # ── Arts (20) ────────────────────────────────────────────────────────
    {"id":"art-01","category":"arts","text":"Beethoven composed his magnificent Ninth Symphony while almost completely deaf."},
    {"id":"art-02","category":"arts","text":"Jazz music originated in New Orleans by blending African rhythms with European harmony."},
    {"id":"art-03","category":"arts","text":"The Impressionist painters captured light and atmosphere with loose, visible brushstrokes."},
    {"id":"art-04","category":"arts","text":"A symphony orchestra typically includes strings, woodwinds, brass, and percussion sections."},
    {"id":"art-05","category":"arts","text":"Shakespeare's plays have been performed continuously around the world for over four centuries."},
    {"id":"art-06","category":"arts","text":"Modern dance breaks away from classical ballet with expressive, free-form movement."},
    {"id":"art-07","category":"arts","text":"Photography transformed how humanity documents and shares visual experiences of the world."},
    {"id":"art-08","category":"arts","text":"Renaissance painters mastered the technique of linear perspective to create convincing depth on flat surfaces."},
    {"id":"art-09","category":"arts","text":"Film editing controls the rhythm and emotional impact of a movie by arranging shots in sequence."},
    {"id":"art-10","category":"arts","text":"The novel as a literary form emerged in the eighteenth century and quickly became a dominant medium."},
    {"id":"art-11","category":"arts","text":"Frida Kahlo's self-portraits explore identity, pain, and cultural heritage through vivid symbolism."},
    {"id":"art-12","category":"arts","text":"Hip-hop culture encompasses rapping, DJing, breakdancing, and graffiti art as interconnected disciplines."},
    {"id":"art-13","category":"arts","text":"Architecture balances aesthetic vision with structural engineering to create functional living spaces."},
    {"id":"art-14","category":"arts","text":"Opera combines orchestral music, vocal performance, theatrical staging, and costume design into a single art form."},
    {"id":"art-15","category":"arts","text":"How did the invention of the camera change the purpose and direction of painting?"},
    {"id":"art-16","category":"arts","text":"What distinguishes a sonata form from a rondo form in classical music composition?"},
    {"id":"art-17","category":"arts","text":"Why do audiences experience such strong emotional responses to minor-key melodies?"},
    {"id":"art-18","category":"arts","text":"How has digital technology expanded the creative possibilities available to visual artists?"},
    {"id":"art-19","category":"arts","text":"What role does improvisation play in the performance of jazz and blues music?"},
    {"id":"art-20","category":"arts","text":"Can a work of art be meaningful even when the viewer knows nothing about its historical context?"},

    # ── Nature (20) ──────────────────────────────────────────────────────
    {"id":"nat-01","category":"nature","text":"The Amazon rainforest produces about twenty percent of the world's atmospheric oxygen."},
    {"id":"nat-02","category":"nature","text":"Coral reefs support more species per unit area than any other marine environment on Earth."},
    {"id":"nat-03","category":"nature","text":"Bird migration patterns span thousands of miles across multiple continents each year."},
    {"id":"nat-04","category":"nature","text":"Volcanic eruptions release gases and molten rock from deep within the Earth's mantle."},
    {"id":"nat-05","category":"nature","text":"Ocean currents play a crucial role in regulating global climate and weather patterns."},
    {"id":"nat-06","category":"nature","text":"Old-growth forests contain complex ecosystems that have developed over many centuries."},
    {"id":"nat-07","category":"nature","text":"The water cycle continuously moves water between the atmosphere, land surfaces, and oceans."},
    {"id":"nat-08","category":"nature","text":"Bees pollinate roughly one-third of the food crops that humans depend on for survival."},
    {"id":"nat-09","category":"nature","text":"Glaciers store about seventy percent of all the fresh water on the planet's surface."},
    {"id":"nat-10","category":"nature","text":"Mangrove forests protect coastlines from erosion and serve as nurseries for juvenile fish."},
    {"id":"nat-11","category":"nature","text":"Soil contains billions of microorganisms per gram that decompose organic matter and recycle nutrients."},
    {"id":"nat-12","category":"nature","text":"The Great Barrier Reef stretches over 2,300 kilometers along the northeastern coast of Australia."},
    {"id":"nat-13","category":"nature","text":"Wolves reintroduced to Yellowstone reshaped the entire ecosystem by altering elk grazing patterns."},
    {"id":"nat-14","category":"nature","text":"Tidal pools form miniature ecosystems where sea stars, anemones, and crabs coexist in shallow rock basins."},
    {"id":"nat-15","category":"nature","text":"How do mycorrhizal fungi form symbiotic networks with tree roots to exchange nutrients underground?"},
    {"id":"nat-16","category":"nature","text":"What causes the rapid decline of insect populations in agricultural regions around the world?"},
    {"id":"nat-17","category":"nature","text":"Why are wetlands considered among the most productive ecosystems on the planet?"},
    {"id":"nat-18","category":"nature","text":"How do deep-ocean hydrothermal vents support life without any energy from sunlight?"},
    {"id":"nat-19","category":"nature","text":"What mechanisms allow certain species to thrive in the extreme cold of the Antarctic?"},
    {"id":"nat-20","category":"nature","text":"Can reforestation projects fully restore the biodiversity of a previously cleared habitat?"},

    # ── History (20) ─────────────────────────────────────────────────────
    {"id":"hist-01","category":"history","text":"The Renaissance saw a dramatic revival of art, science, and classical learning across Europe."},
    {"id":"hist-02","category":"history","text":"The Industrial Revolution transformed manufacturing through steam power and mechanization."},
    {"id":"hist-03","category":"history","text":"Ancient Rome built an extensive network of roads and aqueducts across its vast empire."},
    {"id":"hist-04","category":"history","text":"The invention of the printing press revolutionized the spread of knowledge and literacy."},
    {"id":"hist-05","category":"history","text":"European exploration in the fifteenth century established new trade routes across the globe."},
    {"id":"hist-06","category":"history","text":"The French Revolution fundamentally transformed the political structure of France and Europe."},
    {"id":"hist-07","category":"history","text":"The Silk Road facilitated the exchange of goods, ideas, and culture between East Asia and the Mediterranean."},
    {"id":"hist-08","category":"history","text":"The abolition of slavery in the nineteenth century marked a turning point in the fight for human rights."},
    {"id":"hist-09","category":"history","text":"The construction of the Great Wall of China spanned multiple dynasties and over two thousand years."},
    {"id":"hist-10","category":"history","text":"The Space Race between the United States and the Soviet Union accelerated advances in rocket technology."},
    {"id":"hist-11","category":"history","text":"The signing of the Magna Carta in 1215 established the principle that even monarchs are subject to law."},
    {"id":"hist-12","category":"history","text":"Ancient Egyptian civilization thrived along the Nile for over three thousand years before Roman conquest."},
    {"id":"hist-13","category":"history","text":"The fall of Constantinople in 1453 ended the Byzantine Empire and shifted power in southeastern Europe."},
    {"id":"hist-14","category":"history","text":"The Marshall Plan helped rebuild Western European economies devastated by the Second World War."},
    {"id":"hist-15","category":"history","text":"How did the invention of writing transform governance and record-keeping in ancient Mesopotamia?"},
    {"id":"hist-16","category":"history","text":"What social and economic conditions led to the outbreak of the French Revolution?"},
    {"id":"hist-17","category":"history","text":"Why did the Roman Empire eventually split into eastern and western halves?"},
    {"id":"hist-18","category":"history","text":"How did the Columbian Exchange alter ecosystems and diets on both sides of the Atlantic?"},
    {"id":"hist-19","category":"history","text":"What lessons can modern democracies draw from the political experiments of ancient Athens?"},
    {"id":"hist-20","category":"history","text":"How did the codification of laws in ancient Babylon influence later legal traditions?"},

    # ── Health (20) ──────────────────────────────────────────────────────
    {"id":"health-01","category":"health","text":"Regular aerobic exercise strengthens the heart and reduces the risk of cardiovascular disease."},
    {"id":"health-02","category":"health","text":"A balanced diet rich in fruits, vegetables, and whole grains provides essential vitamins and minerals."},
    {"id":"health-03","category":"health","text":"Chronic sleep deprivation impairs cognitive function and weakens the immune system over time."},
    {"id":"health-04","category":"health","text":"Meditation and deep breathing exercises can measurably reduce stress hormones in the bloodstream."},
    {"id":"health-05","category":"health","text":"Antibiotics treat bacterial infections but are completely ineffective against viral illnesses."},
    {"id":"health-06","category":"health","text":"Exposure to natural sunlight helps the body produce vitamin D, which is critical for bone health."},
    {"id":"health-07","category":"health","text":"Excess sugar consumption contributes to obesity, type 2 diabetes, and chronic inflammation."},
    {"id":"health-08","category":"health","text":"Resistance training preserves bone density and reduces the risk of osteoporosis in older adults."},
    {"id":"health-09","category":"health","text":"Gut microbiome diversity is strongly correlated with overall immune function and mental well-being."},
    {"id":"health-10","category":"health","text":"Handwashing with soap remains the single most effective measure for preventing the spread of infectious diseases."},
    {"id":"health-11","category":"health","text":"Adequate hydration supports kidney function, joint lubrication, and thermoregulation during exercise."},
    {"id":"health-12","category":"health","text":"Cognitive behavioral therapy is one of the most evidence-based treatments for depression and anxiety."},
    {"id":"health-13","category":"health","text":"Omega-3 fatty acids found in fish and flaxseed help reduce systemic inflammation throughout the body."},
    {"id":"health-14","category":"health","text":"Prolonged sitting increases the risk of metabolic syndrome regardless of how much exercise a person does."},
    {"id":"health-15","category":"health","text":"How does the body regulate its core temperature during strenuous physical activity?"},
    {"id":"health-16","category":"health","text":"What role does the placebo effect play in clinical drug trials and patient recovery?"},
    {"id":"health-17","category":"health","text":"Why is antibiotic resistance considered one of the most urgent public health threats today?"},
    {"id":"health-18","category":"health","text":"How do mindfulness practices physically alter neural pathways associated with stress and attention?"},
    {"id":"health-19","category":"health","text":"What dietary changes can help reduce the risk of developing colorectal cancer?"},
    {"id":"health-20","category":"health","text":"Can regular cardiovascular exercise slow the progression of age-related cognitive decline?"},

    # ── Philosophy (20) ──────────────────────────────────────────────────
    {"id":"phil-01","category":"philosophy","text":"Epistemology examines the nature, sources, and limits of human knowledge."},
    {"id":"phil-02","category":"philosophy","text":"The trolley problem illustrates the tension between utilitarian outcomes and deontological principles."},
    {"id":"phil-03","category":"philosophy","text":"Existentialist thinkers argue that individuals must create their own meaning in an indifferent universe."},
    {"id":"phil-04","category":"philosophy","text":"Socrates believed that the unexamined life is not worth living and practiced relentless self-inquiry."},
    {"id":"phil-05","category":"philosophy","text":"Determinism holds that every event is the inevitable result of preceding causes and natural laws."},
    {"id":"phil-06","category":"philosophy","text":"The mind-body problem asks how subjective conscious experience arises from physical brain processes."},
    {"id":"phil-07","category":"philosophy","text":"Utilitarianism judges the morality of an action by the amount of happiness it produces for the greatest number."},
    {"id":"phil-08","category":"philosophy","text":"Immanuel Kant proposed that moral duties are grounded in reason and apply universally to all rational beings."},
    {"id":"phil-09","category":"philosophy","text":"The concept of social contract theory influenced the design of modern democratic constitutions."},
    {"id":"phil-10","category":"philosophy","text":"Phenomenology studies the structures of conscious experience from the first-person point of view."},
    {"id":"phil-11","category":"philosophy","text":"Stoic philosophy teaches that virtue and inner tranquility matter more than external circumstances."},
    {"id":"phil-12","category":"philosophy","text":"The ship of Theseus paradox asks whether an object remains the same when all its parts are gradually replaced."},
    {"id":"phil-13","category":"philosophy","text":"Nihilism rejects the possibility of inherent meaning, purpose, or objective moral values in the world."},
    {"id":"phil-14","category":"philosophy","text":"Pragmatism evaluates ideas and beliefs primarily by their practical consequences and usefulness."},
    {"id":"phil-15","category":"philosophy","text":"Does free will genuinely exist, or is every human decision predetermined by prior brain states?"},
    {"id":"phil-16","category":"philosophy","text":"What distinguishes knowledge from mere true belief in contemporary epistemological theory?"},
    {"id":"phil-17","category":"philosophy","text":"Can a purely logical argument ever settle a fundamentally ethical disagreement between two people?"},
    {"id":"phil-18","category":"philosophy","text":"Is consciousness an emergent property of complex neural networks, or something more fundamental?"},
    {"id":"phil-19","category":"philosophy","text":"How should society balance individual liberty against the collective welfare of its citizens?"},
    {"id":"phil-20","category":"philosophy","text":"What obligations, if any, do present generations owe to people who have not yet been born?"},

    # ── Business (20) ────────────────────────────────────────────────────
    {"id":"biz-01","category":"business","text":"Venture capital firms invest in early-stage startups in exchange for equity and board representation."},
    {"id":"biz-02","category":"business","text":"Supply chain disruptions during global crises can cascade rapidly across interconnected industries."},
    {"id":"biz-03","category":"business","text":"Compound interest allows invested capital to grow exponentially over long periods of time."},
    {"id":"biz-04","category":"business","text":"Effective marketing communicates a product's value proposition clearly to its target audience."},
    {"id":"biz-05","category":"business","text":"Diversifying an investment portfolio reduces overall risk by spreading capital across uncorrelated assets."},
    {"id":"biz-06","category":"business","text":"Intellectual property protections like patents and trademarks incentivize innovation by safeguarding creators."},
    {"id":"biz-07","category":"business","text":"Remote work has fundamentally altered office culture, commuting patterns, and urban real estate demand."},
    {"id":"biz-08","category":"business","text":"Inflation erodes purchasing power and disproportionately affects households with fixed incomes."},
    {"id":"biz-09","category":"business","text":"A clear mission statement aligns an organization's strategy, culture, and decision-making processes."},
    {"id":"biz-10","category":"business","text":"Central banks adjust interest rates to influence borrowing, spending, and overall economic growth."},
    {"id":"biz-11","category":"business","text":"Economies of scale reduce the per-unit cost of production as a company increases its output volume."},
    {"id":"biz-12","category":"business","text":"Corporate social responsibility programs aim to generate positive environmental and community impact alongside profit."},
    {"id":"biz-13","category":"business","text":"Mergers and acquisitions allow companies to rapidly expand their market share and capabilities."},
    {"id":"biz-14","category":"business","text":"Consumer behavior research reveals how psychological biases influence purchasing decisions."},
    {"id":"biz-15","category":"business","text":"How do central banks decide when to raise or lower benchmark interest rates?"},
    {"id":"biz-16","category":"business","text":"What factors determine whether a startup will succeed or fail in its first three years?"},
    {"id":"biz-17","category":"business","text":"Why do some industries tend naturally toward monopoly while others remain fiercely competitive?"},
    {"id":"biz-18","category":"business","text":"How has globalization shifted manufacturing employment from developed to developing economies?"},
    {"id":"biz-19","category":"business","text":"What ethical responsibilities do technology companies have regarding user data and privacy?"},
    {"id":"biz-20","category":"business","text":"Can universal basic income programs sustain economic productivity while reducing poverty?"},
]

QUERIES = [
    {"id":"q-gravity",       "text":"How does gravity work in the vacuum of outer space?"},
    {"id":"q-recipes",       "text":"What recipes should I try for a weeknight dinner?"},
    {"id":"q-musicians",     "text":"Tell me about famous classical musicians and their compositions."},
    {"id":"q-fitness",       "text":"How can I improve my physical fitness and overall health?"},
    {"id":"q-ai-econ",       "text":"What are the economic impacts of artificial intelligence on global labor markets?"},
    {"id":"q-ecosystems",    "text":"How do ecosystems recover after large-scale natural disasters?"},
    {"id":"q-ethics-gene",   "text":"What ethical questions does genetic engineering raise for society?"},
    {"id":"q-ancient-arch",  "text":"How did ancient civilizations influence modern architectural design?"},
]


def main():
    print(f"Loading model all-MiniLM-L6-v2...", file=sys.stderr)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [s["text"] for s in SENTENCES]
    query_texts = [q["text"] for q in QUERIES]
    print(f"Encoding {len(texts)} sentences + {len(query_texts)} queries...", file=sys.stderr)

    all_texts = texts + query_texts
    all_embeddings = model.encode(all_texts, show_progress_bar=False)
    embeddings = all_embeddings[:len(texts)]
    query_embeddings = all_embeddings[len(texts):]

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
