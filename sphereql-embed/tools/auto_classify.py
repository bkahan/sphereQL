#!/usr/bin/env python3
"""Encode sentences AND auto-categorize them with zero-shot NLI classification.

Uses:
  - all-MiniLM-L6-v2 for sentence embeddings (384-d)
  - facebook/bart-large-mnli for zero-shot category inference

No pre-assigned categories. The model decides.

Usage:
    python3 auto_classify.py                       # writes JSON to stdout
    python3 auto_classify.py > embeddings.json     # save to file
"""

import json
import sys
from collections import Counter
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

# ── Candidate labels for zero-shot classification ────────────────────────
# These are broad topic areas. The classifier picks the best match per
# sentence without ever seeing training examples for these labels.
CANDIDATE_LABELS = [
    "science",
    "technology",
    "sports",
    "cooking",
    "arts",
    "nature",
    "history",
    "health",
    "philosophy",
    "business",
]

# ── Sentences (no pre-assigned categories) ───────────────────────────────
SENTENCES = [
    # -- batch 1: physics, chemistry, biology --
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "DNA carries the genetic instructions for the development of all known living organisms.",
    "Quantum entanglement allows particles to be correlated instantaneously across vast distances.",
    "Einstein's theory of general relativity describes gravity as the curvature of spacetime.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
    "The periodic table organizes chemical elements by their atomic number and electron configuration.",
    "Mitochondria generate most of the cell's supply of adenosine triphosphate through oxidative phosphorylation.",
    "The Higgs boson gives other fundamental particles their mass through interaction with the Higgs field.",
    "Plate tectonics explains how the Earth's lithosphere is divided into large plates that slowly move.",
    "Entropy in a closed system tends to increase over time according to the second law of thermodynamics.",
    "The double-helix structure of DNA was first described by Watson and Crick in 1953.",
    "Neutrinos are nearly massless subatomic particles that rarely interact with ordinary matter.",
    "How does the theory of evolution explain the diversity of life on Earth?",
    "What causes the northern lights to appear in polar regions of the sky?",
    "Why does the uncertainty principle place a fundamental limit on measuring quantum systems?",
    "How do vaccines train the immune system to recognize and fight specific pathogens?",
    "What evidence supports the theory that the universe originated from the Big Bang?",
    "Can stem cells be reliably programmed to regenerate damaged tissues in the human body?",

    # -- batch 2: computing, AI, engineering --
    "Machine learning algorithms improve their performance by training on large datasets.",
    "The internet connects billions of devices through a global network of computer systems.",
    "Artificial intelligence can now generate realistic images and human-like text responses.",
    "Blockchain technology provides a decentralized and tamper-resistant digital ledger system.",
    "Autonomous vehicles use sensors and neural networks to navigate roads without human input.",
    "Cloud computing allows on-demand access to shared computing resources over the internet.",
    "Quantum computers use qubits to solve certain problems exponentially faster than classical machines.",
    "Cybersecurity protects computer systems and networks from digital attacks and data breaches.",
    "Natural language processing enables computers to understand and generate human language.",
    "The global positioning system uses a constellation of satellites to determine precise locations on Earth.",
    "Three-dimensional printing builds physical objects layer by layer from digital blueprints.",
    "Edge computing processes data closer to where it is generated rather than in a centralized data center.",
    "Reinforcement learning agents discover optimal strategies through trial and error in simulated environments.",
    "The Internet of Things connects everyday appliances and industrial equipment to the web.",
    "How do convolutional neural networks recognize objects in photographs?",
    "What security risks does widespread adoption of smart home devices introduce?",
    "Can artificial general intelligence ever match the flexibility of human reasoning?",
    "How does end-to-end encryption protect private messages from being intercepted?",
    "What role do graphics processing units play in accelerating deep learning training?",
    "Why is version control essential for collaborative software development projects?",

    # -- batch 3: athletics, fitness, competition --
    "The Olympic Games bring together athletes from around the world to compete every four years.",
    "A marathon covers 26.2 miles and tests the endurance limits of long-distance runners.",
    "Basketball requires teamwork, agility, and precise shooting to outscore the opposing team.",
    "Soccer is the most widely followed sport in the world, with over four billion fans.",
    "Tennis matches can last for hours as players rally the ball across the net.",
    "Swimming is an excellent full-body cardiovascular exercise suitable for people of all ages.",
    "Professional athletes dedicate years of intense training to compete at the highest level.",
    "A well-designed strength training program builds muscle, strengthens joints, and prevents injuries.",
    "Ice hockey combines skating speed, stick handling, and physical contact in a fast-paced arena.",
    "Gymnastics demands extraordinary flexibility, balance, and explosive power from its competitors.",
    "Cricket matches can unfold over five days in the traditional test format of the game.",
    "Rock climbing challenges both physical strength and mental problem-solving on vertical terrain.",
    "Interval training alternates bursts of high-intensity effort with periods of active recovery.",
    "The Tour de France is the most prestigious multi-stage cycling race in professional road cycling.",
    "How does altitude training improve an athlete's oxygen-carrying capacity?",
    "What strategies help prevent overuse injuries in endurance athletes?",
    "Why is proper hydration so critical for sustaining peak athletic performance?",
    "How do referees use video replay technology to make fairer officiating decisions?",
    "What mental techniques do elite competitors use to stay focused under pressure?",
    "Can data analytics give sports teams a measurable competitive advantage?",

    # -- batch 4: recipes, techniques, ingredients --
    "Preheat the oven to 375 degrees and line the baking sheet with parchment paper.",
    "Fresh herbs like basil and cilantro add vibrant flavor to almost any savory dish.",
    "A good risotto requires patience and constant stirring to achieve a creamy texture.",
    "Marinating meat overnight allows the flavors to penetrate deeply into the protein.",
    "The secret to a perfect pie crust is using very cold butter and minimal handling of the dough.",
    "Sauteing onions slowly until caramelized brings out their natural sweetness.",
    "Homemade pasta has a remarkably different taste and texture from store-bought dried varieties.",
    "A cast-iron skillet retains heat evenly and develops a natural nonstick surface over time.",
    "Fermented foods like kimchi and sauerkraut support healthy digestion with beneficial bacteria.",
    "Reducing a stock over low heat concentrates its flavor into a rich, glossy sauce.",
    "Sourdough bread relies on a live culture of wild yeast and lactic acid bacteria for leavening.",
    "Blanching vegetables in boiling water and then plunging them into ice water preserves their color.",
    "A sharp knife is safer than a dull one because it requires less force and slips less often.",
    "Emulsification binds oil and vinegar together with the help of an agent like egg yolk or mustard.",
    "How does the Maillard reaction create the crispy brown crust on seared steaks?",
    "What makes sourdough fermentation produce a tangier flavor than commercial yeast breads?",
    "Why does tempering chocolate require precise temperature control to achieve a glossy finish?",
    "Which spices pair best with roasted root vegetables for a hearty autumn side dish?",
    "How can home cooks safely preserve seasonal produce through canning and pickling?",
    "What techniques produce the flakiest layers in laminated pastry doughs like croissants?",

    # -- batch 5: music, painting, literature, film --
    "Beethoven composed his magnificent Ninth Symphony while almost completely deaf.",
    "Jazz music originated in New Orleans by blending African rhythms with European harmony.",
    "The Impressionist painters captured light and atmosphere with loose, visible brushstrokes.",
    "A symphony orchestra typically includes strings, woodwinds, brass, and percussion sections.",
    "Shakespeare's plays have been performed continuously around the world for over four centuries.",
    "Modern dance breaks away from classical ballet with expressive, free-form movement.",
    "Photography transformed how humanity documents and shares visual experiences of the world.",
    "Renaissance painters mastered the technique of linear perspective to create convincing depth.",
    "Film editing controls the rhythm and emotional impact of a movie by arranging shots in sequence.",
    "The novel as a literary form emerged in the eighteenth century and quickly became dominant.",
    "Frida Kahlo's self-portraits explore identity, pain, and cultural heritage through vivid symbolism.",
    "Hip-hop culture encompasses rapping, DJing, breakdancing, and graffiti art as interconnected disciplines.",
    "Architecture balances aesthetic vision with structural engineering to create functional living spaces.",
    "Opera combines orchestral music, vocal performance, theatrical staging, and costume design.",
    "How did the invention of the camera change the purpose and direction of painting?",
    "What distinguishes a sonata form from a rondo form in classical music composition?",
    "Why do audiences experience such strong emotional responses to minor-key melodies?",
    "How has digital technology expanded the creative possibilities available to visual artists?",
    "What role does improvisation play in the performance of jazz and blues music?",
    "Can a work of art be meaningful even when the viewer knows nothing about its context?",

    # -- batch 6: ecology, geology, weather, oceans --
    "The Amazon rainforest produces about twenty percent of the world's atmospheric oxygen.",
    "Coral reefs support more species per unit area than any other marine environment on Earth.",
    "Bird migration patterns span thousands of miles across multiple continents each year.",
    "Volcanic eruptions release gases and molten rock from deep within the Earth's mantle.",
    "Ocean currents play a crucial role in regulating global climate and weather patterns.",
    "Old-growth forests contain complex ecosystems that have developed over many centuries.",
    "The water cycle continuously moves water between the atmosphere, land surfaces, and oceans.",
    "Bees pollinate roughly one-third of the food crops that humans depend on for survival.",
    "Glaciers store about seventy percent of all the fresh water on the planet's surface.",
    "Mangrove forests protect coastlines from erosion and serve as nurseries for juvenile fish.",
    "Soil contains billions of microorganisms per gram that decompose organic matter and recycle nutrients.",
    "The Great Barrier Reef stretches over 2,300 kilometers along the coast of Australia.",
    "Wolves reintroduced to Yellowstone reshaped the entire ecosystem by altering elk grazing patterns.",
    "Tidal pools form miniature ecosystems where sea stars, anemones, and crabs coexist.",
    "How do mycorrhizal fungi form symbiotic networks with tree roots to exchange nutrients?",
    "What causes the rapid decline of insect populations in agricultural regions?",
    "Why are wetlands considered among the most productive ecosystems on the planet?",
    "How do deep-ocean hydrothermal vents support life without any energy from sunlight?",
    "What mechanisms allow certain species to thrive in the extreme cold of the Antarctic?",
    "Can reforestation projects fully restore the biodiversity of a previously cleared habitat?",

    # -- batch 7: civilizations, revolutions, exploration --
    "The Renaissance saw a dramatic revival of art, science, and classical learning across Europe.",
    "The Industrial Revolution transformed manufacturing through steam power and mechanization.",
    "Ancient Rome built an extensive network of roads and aqueducts across its vast empire.",
    "The invention of the printing press revolutionized the spread of knowledge and literacy.",
    "European exploration in the fifteenth century established new trade routes across the globe.",
    "The French Revolution fundamentally transformed the political structure of France and Europe.",
    "The Silk Road facilitated the exchange of goods, ideas, and culture between East Asia and the Mediterranean.",
    "The abolition of slavery in the nineteenth century marked a turning point for human rights.",
    "The construction of the Great Wall of China spanned multiple dynasties over two thousand years.",
    "The Space Race between the United States and the Soviet Union accelerated rocket technology.",
    "The signing of the Magna Carta in 1215 established that even monarchs are subject to law.",
    "Ancient Egyptian civilization thrived along the Nile for over three thousand years.",
    "The fall of Constantinople in 1453 ended the Byzantine Empire and shifted European power.",
    "The Marshall Plan helped rebuild Western European economies after the Second World War.",
    "How did the invention of writing transform governance in ancient Mesopotamia?",
    "What social and economic conditions led to the outbreak of the French Revolution?",
    "Why did the Roman Empire eventually split into eastern and western halves?",
    "How did the Columbian Exchange alter ecosystems and diets across the Atlantic?",
    "What lessons can modern democracies draw from the political experiments of ancient Athens?",
    "How did the codification of laws in ancient Babylon influence later legal traditions?",

    # -- batch 8: medicine, wellness, nutrition --
    "Regular aerobic exercise strengthens the heart and reduces the risk of cardiovascular disease.",
    "A balanced diet rich in fruits, vegetables, and whole grains provides essential vitamins.",
    "Chronic sleep deprivation impairs cognitive function and weakens the immune system over time.",
    "Meditation and deep breathing exercises can measurably reduce stress hormones in the bloodstream.",
    "Antibiotics treat bacterial infections but are completely ineffective against viral illnesses.",
    "Exposure to natural sunlight helps the body produce vitamin D, which is critical for bone health.",
    "Excess sugar consumption contributes to obesity, type 2 diabetes, and chronic inflammation.",
    "Resistance training preserves bone density and reduces the risk of osteoporosis in older adults.",
    "Gut microbiome diversity is strongly correlated with overall immune function and mental well-being.",
    "Handwashing with soap remains the most effective measure for preventing the spread of disease.",
    "Adequate hydration supports kidney function, joint lubrication, and thermoregulation during exercise.",
    "Cognitive behavioral therapy is one of the most evidence-based treatments for depression and anxiety.",
    "Omega-3 fatty acids found in fish and flaxseed help reduce systemic inflammation.",
    "Prolonged sitting increases the risk of metabolic syndrome regardless of exercise habits.",
    "How does the body regulate its core temperature during strenuous physical activity?",
    "What role does the placebo effect play in clinical drug trials and patient recovery?",
    "Why is antibiotic resistance considered one of the most urgent public health threats today?",
    "How do mindfulness practices physically alter neural pathways associated with stress?",
    "What dietary changes can help reduce the risk of developing colorectal cancer?",
    "Can regular cardiovascular exercise slow the progression of age-related cognitive decline?",

    # -- batch 9: ethics, consciousness, logic, meaning --
    "Epistemology examines the nature, sources, and limits of human knowledge.",
    "The trolley problem illustrates the tension between utilitarian outcomes and deontological duties.",
    "Existentialist thinkers argue that individuals must create their own meaning in an indifferent universe.",
    "Socrates believed that the unexamined life is not worth living and practiced relentless self-inquiry.",
    "Determinism holds that every event is the inevitable result of preceding causes and natural laws.",
    "The mind-body problem asks how subjective conscious experience arises from physical brain processes.",
    "Utilitarianism judges the morality of an action by the happiness it produces for the greatest number.",
    "Immanuel Kant proposed that moral duties are grounded in reason and apply universally.",
    "The concept of social contract theory influenced the design of modern democratic constitutions.",
    "Phenomenology studies the structures of conscious experience from the first-person point of view.",
    "Stoic philosophy teaches that virtue and inner tranquility matter more than external circumstances.",
    "The ship of Theseus paradox asks whether an object remains the same when all its parts are replaced.",
    "Nihilism rejects the possibility of inherent meaning, purpose, or objective moral values.",
    "Pragmatism evaluates ideas and beliefs primarily by their practical consequences and usefulness.",
    "Does free will genuinely exist, or is every human decision predetermined by prior brain states?",
    "What distinguishes knowledge from mere true belief in contemporary epistemological theory?",
    "Can a purely logical argument ever settle a fundamentally ethical disagreement?",
    "Is consciousness an emergent property of complex neural networks, or something more fundamental?",
    "How should society balance individual liberty against the collective welfare of its citizens?",
    "What obligations, if any, do present generations owe to people who have not yet been born?",

    # -- batch 10: economics, entrepreneurship, finance --
    "Venture capital firms invest in early-stage startups in exchange for equity and board seats.",
    "Supply chain disruptions during global crises can cascade rapidly across interconnected industries.",
    "Compound interest allows invested capital to grow exponentially over long periods of time.",
    "Effective marketing communicates a product's value proposition clearly to its target audience.",
    "Diversifying an investment portfolio reduces overall risk by spreading capital across assets.",
    "Intellectual property protections like patents and trademarks incentivize innovation.",
    "Remote work has fundamentally altered office culture, commuting patterns, and real estate demand.",
    "Inflation erodes purchasing power and disproportionately affects households with fixed incomes.",
    "A clear mission statement aligns an organization's strategy, culture, and decision-making.",
    "Central banks adjust interest rates to influence borrowing, spending, and economic growth.",
    "Economies of scale reduce the per-unit cost of production as output volume increases.",
    "Corporate social responsibility programs aim to generate positive impact alongside profit.",
    "Mergers and acquisitions allow companies to rapidly expand their market share and capabilities.",
    "Consumer behavior research reveals how psychological biases influence purchasing decisions.",
    "How do central banks decide when to raise or lower benchmark interest rates?",
    "What factors determine whether a startup will succeed or fail in its first three years?",
    "Why do some industries tend naturally toward monopoly while others remain competitive?",
    "How has globalization shifted manufacturing employment from developed to developing economies?",
    "What ethical responsibilities do technology companies have regarding user data and privacy?",
    "Can universal basic income programs sustain economic productivity while reducing poverty?",

    # -- batch 11: cross-domain bridges --
    "How did advances in physics lead to the development of nuclear energy and its controversies?",
    "What nutritional strategies do Olympic athletes follow to enhance their performance?",
    "How has artificial intelligence changed the way we compose and produce music?",
    "What role do forest ecosystems play in mitigating the effects of climate change?",
    "How do cultural and historical traditions shape modern cooking techniques around the world?",
    "What philosophical questions arise from the possibility of uploading human consciousness to a computer?",
    "How do economic incentives affect the pace of renewable energy adoption in developing nations?",
    "What parallels exist between the structure of biological neural networks and artificial ones?",
    "How has the commercialization of space exploration changed the aerospace industry?",
    "What ethical dilemmas do autonomous weapons systems create for international law?",
    "How does urban planning affect both public health outcomes and local economic growth?",
    "What connections exist between musical training in childhood and later academic performance?",
    "How do traditional fermentation techniques used in cooking relate to modern biotechnology?",
    "What impact has social media had on political movements and democratic participation?",
    "How do psychological insights from philosophy inform the design of user interfaces in technology?",
    "What role does clean water access play in both public health and economic development?",
    "How has the discovery of ancient trade routes reshaped our understanding of cultural exchange?",
    "What lessons from sports psychology can be applied to workplace performance and leadership?",
    "How do advances in materials science enable new forms of architectural design?",
    "What connections exist between biodiversity loss and the stability of global food systems?",

    # -- batch 12: additional depth --
    "Gravitational waves were first directly detected by the LIGO observatory in September 2015.",
    "The human genome project mapped approximately 20,500 protein-coding genes in human DNA.",
    "Superconductors conduct electricity with zero resistance below a critical temperature.",
    "CRISPR gene editing technology allows precise modifications to the DNA of living organisms.",
    "Dark matter makes up roughly 27 percent of the total mass-energy content of the universe.",
    "Functional magnetic resonance imaging measures brain activity by detecting changes in blood flow.",
    "The standard model of particle physics describes three of the four fundamental forces of nature.",
    "Epigenetic modifications can alter gene expression without changing the underlying DNA sequence.",
    "How do prions cause disease if they contain no DNA or RNA of their own?",
    "What is the current scientific consensus on the age and eventual fate of the universe?",
    "Electric vehicles are rapidly gaining market share as battery technology costs continue to fall.",
    "Large language models process text by predicting the most probable next token in a sequence.",
    "Containerization with tools like Docker has transformed how software is deployed and scaled.",
    "Augmented reality overlays digital information onto the physical world through wearable devices.",
    "The development of mRNA vaccine technology was accelerated by decades of prior research.",
    "Microplastics have been found in ocean sediments, drinking water, and even human blood samples.",
    "Coral bleaching occurs when rising ocean temperatures stress the symbiotic algae living in corals.",
    "Permafrost thawing in Arctic regions releases methane, a potent greenhouse gas.",
    "How does deforestation in tropical regions affect rainfall patterns thousands of miles away?",
    "What are the long-term ecological consequences of introducing invasive species to island ecosystems?",
    "Stoic practices like negative visualization are experiencing a revival in modern self-help literature.",
    "Behavioral economics combines psychological insights with classical economic models of decision-making.",
    "The gig economy has created new forms of flexible employment but also new forms of worker insecurity.",
    "Venture debt provides growth capital to startups without diluting existing shareholders.",
    "How do cultural attitudes toward failure affect rates of entrepreneurship across different countries?",
    "What role does sleep play in consolidating memories and supporting long-term learning?",
    "Progressive overload is the fundamental principle behind all effective strength training programs.",
    "High-intensity interval training can produce cardiovascular benefits comparable to longer steady-state sessions.",
    "The Mediterranean diet has been consistently linked to lower rates of heart disease and longer lifespan.",
    "What are the most effective evidence-based strategies for quitting smoking permanently?",

    # -- batch 13: more questions and depth --
    "Why do some volcanic eruptions produce explosive ash clouds while others flow as rivers of lava?",
    "How did the development of refrigeration technology transform global food distribution?",
    "What mechanisms allow migratory animals to navigate accurately across thousands of miles?",
    "How do noise pollution and light pollution affect wildlife behavior in urban environments?",
    "What is the relationship between gut bacteria composition and mental health conditions like depression?",
    "How do central bank digital currencies differ from decentralized cryptocurrencies like Bitcoin?",
    "What cognitive biases most commonly lead investors to make irrational financial decisions?",
    "How did the printing of books in vernacular languages accelerate the Protestant Reformation?",
    "What evidence suggests that regular exposure to nature improves mental health and reduces stress?",
    "How do trade agreements between nations affect domestic employment and wage levels?",
    "Why is the concept of falsifiability central to the scientific method proposed by Karl Popper?",
    "What distinguishes abstract expressionism from earlier movements in Western art history?",
    "How do tidal energy systems convert the motion of ocean tides into usable electricity?",
    "What psychological factors explain why eyewitness testimony is often unreliable in courtrooms?",
    "How has the rise of streaming platforms changed the economics of the music industry?",
    "What are the potential benefits and risks of using nuclear fusion as a future energy source?",
    "How do mangrove restoration projects contribute to both carbon sequestration and coastal protection?",
    "What factors determine the carrying capacity of an ecosystem for a given species?",
    "How did the construction of transcontinental railroads reshape economies in the nineteenth century?",
    "What are the ethical implications of using predictive algorithms in criminal sentencing decisions?",
]

QUERIES = [
    "How does gravity work in the vacuum of outer space?",
    "What recipes should I try for a weeknight dinner?",
    "Tell me about famous classical musicians and their compositions.",
    "How can I improve my physical fitness and overall health?",
    "What are the economic impacts of artificial intelligence on global labor markets?",
    "How do ecosystems recover after large-scale natural disasters?",
    "What ethical questions does genetic engineering raise for society?",
    "How did ancient civilizations influence modern architectural design?",
]


def main():
    n = len(SENTENCES)
    print(f"Loading models...", file=sys.stderr)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    classifier = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,  # CPU
    )

    # ── Encode ───────────────────────────────────────────────────────────
    print(f"Encoding {n} sentences + {len(QUERIES)} queries...", file=sys.stderr)
    all_texts = SENTENCES + QUERIES
    all_embeddings = embedder.encode(all_texts, show_progress_bar=True)
    sentence_embeddings = all_embeddings[:n]
    query_embeddings = all_embeddings[n:]

    # ── Classify ─────────────────────────────────────────────────────────
    print(f"Classifying {n} sentences (zero-shot NLI)...", file=sys.stderr)
    batch_size = 16
    categories = []
    for i in range(0, n, batch_size):
        batch = SENTENCES[i : i + batch_size]
        results = classifier(batch, candidate_labels=CANDIDATE_LABELS, batch_size=batch_size)
        if isinstance(results, dict):
            results = [results]
        for r in results:
            categories.append(r["labels"][0])
        done = min(i + batch_size, n)
        print(f"  classified {done}/{n}", file=sys.stderr)

    # ── Assemble output ──────────────────────────────────────────────────
    output = {
        "model": "all-MiniLM-L6-v2",
        "classifier": "facebook/bart-large-mnli",
        "dimension": int(sentence_embeddings.shape[1]),
        "candidate_labels": CANDIDATE_LABELS,
        "sentences": [],
        "queries": [],
    }

    for i, (text, cat, emb) in enumerate(
        zip(SENTENCES, categories, sentence_embeddings)
    ):
        output["sentences"].append(
            {
                "id": f"s-{i:03d}",
                "text": text,
                "category": cat,
                "embedding": [round(float(x), 6) for x in emb],
            }
        )

    for i, (text, emb) in enumerate(zip(QUERIES, query_embeddings)):
        output["queries"].append(
            {
                "id": f"q-{i:03d}",
                "text": text,
                "embedding": [round(float(x), 6) for x in emb],
            }
        )

    json.dump(output, sys.stdout, indent=None, separators=(",", ":"))
    sys.stdout.flush()

    # ── Summary ──────────────────────────────────────────────────────────
    counts = Counter(categories)
    print(f"\nDone: {n} sentences, {len(QUERIES)} queries, {sentence_embeddings.shape[1]}-d", file=sys.stderr)
    print(f"Category distribution (auto-classified):", file=sys.stderr)
    for cat, cnt in counts.most_common():
        print(f"  {cat:<14} {cnt}", file=sys.stderr)


if __name__ == "__main__":
    main()
