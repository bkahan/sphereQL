"""
Mapping tables for the SphereQL corpus generator.

Three core tables:
- KEYWORD_TO_AXIS: keyword substring → axis index (0–127)
- FIELD_TO_CATEGORY / FIELD_MULTI_MAP: OpenAlex field ID → SphereQL category
- CATEGORY_PRIMARY_AXES: SphereQL category → list of primary axis indices
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════
#  KEYWORD → AXIS INDEX
#
#  Keys are lowercase substrings matched against OpenAlex topic keywords,
#  display_name, and description. Values are axis indices matching
#  sphereql-corpus/src/axes.rs. Longer keys should appear before shorter
#  ones for any prefix conflicts (the generator matches all, so order
#  within the dict doesn't affect correctness — but it aids readability).
# ═══════════════════════════════════════════════════════════════════════

KEYWORD_TO_AXIS: dict[str, int] = {
    # ── Physics (0–6) ──
    "energy": 0, "thermodynamic": 0, "kinetic": 0, "thermal": 0,
    "heat transfer": 0, "exotherm": 0, "endotherm": 0, "calorimetr": 0,
    "force": 1, "gravity": 1, "gravitational": 1, "newton": 1,
    "pressure": 1, "tension": 1, "friction": 1, "momentum": 1, "torque": 1,
    "quantum": 2, "qubit": 2, "superposition": 2, "entangle": 2,
    "planck": 2, "heisenberg": 2, "wave function": 2, "tunneling": 2, "decoherence": 2,
    "wave": 3, "oscillat": 3, "frequency": 3, "amplitude": 3,
    "wavelength": 3, "vibrat": 3, "resonan": 3, "interferen": 3, "diffract": 3,
    "entropy": 4, "disorder": 4, "irreversib": 4, "second law": 4, "boltzmann": 4,
    "relativity": 5, "einstein": 5, "spacetime": 5, "lorentz": 5,
    "time dilation": 5, "gravitational wave": 5, "geodesic": 5,
    "particle": 6, "hadron": 6, "lepton": 6, "boson": 6,
    "fermion": 6, "quark": 6, "gluon": 6, "photon": 6, "neutrino": 6,
    "muon": 6, "collider": 6, "accelerator": 6,

    # ── Mathematics (7–11) ──
    "mathematic": 7, "theorem": 7, "equation": 7, "numerical": 7,
    "formul": 7, "conjecture": 7, "axiom": 7,
    "proof": 8, "formal verif": 8, "deduction": 8, "induction": 8, "lemma": 8,
    "calculus": 9, "differential": 9, "integral": 9, "derivative": 9,
    "gradient": 9, "laplacian": 9, "variational": 9, "stochastic calcul": 9,
    "graph theory": 10, "vertex": 10, "edge coloring": 10, "planar graph": 10,
    "combinatoric": 10, "network theory": 10, "tree structure": 10,
    "algebra": 11, "group theory": 11, "ring theory": 11, "linear algebra": 11,
    "matrix": 11, "vector space": 11, "eigenvalue": 11, "tensor": 11,
    "polynomial": 11, "homomorphism": 11,

    # ── Biology (12–15) ──
    "life": 12, "living": 12, "organism": 12, "biological": 12, "biolog": 12,
    "biodiversity": 12, "species": 12, "flora": 12, "fauna": 12,
    "evolution": 13, "natural selection": 13, "adaptation": 13, "phylogenet": 13,
    "speciation": 13, "darwin": 13, "mutation rate": 13,
    "genetic": 14, "genome": 14, "dna": 14, "chromosom": 14,
    "gene expression": 14, "crispr": 14, "heredit": 14, "allele": 14,
    "epigenet": 14, "genotype": 14, "phenotype": 14, "transgenic": 14,
    "messenger rna": 14, "non-coding rna": 14, "rna sequenc": 14,
    "small interfering rna": 14, "micro rna": 14, "rna polymerase": 14,
    "cell": 15, "cellular": 15, "mitosis": 15, "meiosis": 15,
    "cytoplasm": 15, "membrane": 15, "organelle": 15, "apoptosis": 15,
    "stem cell": 15, "proliferat": 15,

    # ── Chemistry (16–18) ──
    "chemistry": 16, "chemical": 16, "compound": 16, "element": 16,
    "periodic table": 16, "stoichiometr": 16, "titrat": 16,
    "molecule": 17, "molecular": 17, "polymer": 17, "protein": 17,
    "enzyme": 17, "amino acid": 17, "peptide": 17, "lipid": 17,
    "nucleotide": 17, "macromolecul": 17, "biomolecul": 17,
    "reaction": 18, "catalys": 18, "oxidat": 18, "reduct": 18,
    "synthesis": 18, "reagent": 18, "equilibrium": 18,
    "electrochemis": 18, "photochem": 18, "combustion": 18,

    # ── Medicine (19–22) ──
    "diagnos": 19, "screening": 19, "imaging": 19, "biomarker": 19,
    "mri": 19, "ct scan": 19, "x-ray": 19, "biopsy": 19, "prognos": 19,
    "therap": 20, "treatment": 20, "drug": 20, "pharmacolog": 20,
    "dosage": 20, "clinical trial": 20, "intervention": 20, "rehabilitat": 20,
    "surgery": 20, "radiation therapy": 20, "chemotherapy": 20,
    "anatomy": 21, "organs": 21, "organ system": 21, "organ transplant": 21,
    "internal organ": 21, "tissue": 21, "bone": 21, "muscle": 21,
    "vascular": 21, "cardiac": 21, "pulmonar": 21, "renal": 21,
    "hepat": 21, "gastrointestin": 21, "skeletal": 21,
    "clinical": 22, "patient": 22, "hospital": 22, "epidemiolog": 22,
    "morbidity": 22, "mortality": 22, "comorbid": 22, "symptom": 22,
    "patholog": 22, "oncolog": 22, "chronic disease": 22, "acute care": 22,

    # ── Neuroscience (23–25) ──
    "neural": 23, "neuron": 23, "synap": 23, "dendrit": 23,
    "neurotransmit": 23, "cortical": 23, "dopamin": 23, "serotonin": 23,
    "neuroplastic": 23, "electrophysiolog": 23,
    "axonal": 23, "axon terminal": 23, "axon hillock": 23, "myelinated axon": 23,
    "brain": 24, "cerebr": 24, "hippocampus": 24, "amygdala": 24,
    "prefrontal": 24, "thalamus": 24, "basal ganglia": 24,
    "cortex": 24, "white matter": 24, "grey matter": 24, "fmri": 24, "eeg": 24,
    "consciousness": 25, "conscious": 25, "awareness": 25, "qualia": 25,
    "subjective experience": 25, "phenomenal": 25, "sentien": 25,

    # ── Computer Science (26–29) ──
    "comput": 26, "processor": 26, "cpu": 26, "hardware": 26,
    "turing": 26, "automata": 26, "complexity": 26, "computab": 26,
    "parallel comput": 26, "distributed comput": 26,
    "boolean": 27, "propositional": 27, "predicate": 27,
    "satisfiab": 27, "decidab": 27, "modal logic": 27, "fuzzy logic": 27,
    "formal logic": 27, "symbolic logic": 27, "first-order logic": 27,
    "logic programming": 27, "logic gate": 27, "logic circuit": 27,
    "software": 28, "programming": 28, "compiler": 28, "debug": 28,
    "operating system": 28, "version control": 28,
    "devops": 28, "microservice": 28, "refactor": 28, "codebase": 28,
    "api endpoint": 28, "rest api": 28, "graphql api": 28, "api design": 28,
    "web api": 28, "api gateway": 28,
    "algorithm": 29, "sorting": 29, "searching": 29, "hashing": 29,
    "dynamic programming": 29, "greedy": 29, "divide and conquer": 29,
    "recursion": 29, "big-o": 29, "time complexity": 29,

    # ── AI / Data Science (30–33) ──
    "artificial intelligen": 30, " ai ": 30, "intelligent agent": 30,
    "expert system": 30, "knowledge represent": 30,
    "computer vision": 30, "robotics": 30, "autonomous": 30,
    "large language model": 31, "llm": 31, "transformer": 31, "gpt": 31,
    "attention mechanism": 31, "fine-tun": 31,
    "generative ai": 31, "foundation model": 31, "chatbot": 31,
    "bert model": 31, "bert language": 31, "bidirectional encoder": 31,
    "pretrained language": 31,
    "data": 32, "dataset": 32, "database": 32, "data mining": 32,
    "big data": 32, "data warehouse": 32, "data pipeline": 32,
    "sql": 32, "nosql": 32, "data governance": 32, "data etl": 32, "etl pipeline": 32,
    "machine learn": 33, "deep learn": 33, "neural network": 33,
    "supervised": 33, "unsupervised": 33, "reinforcement learn": 33,
    "classification": 33, "clustering": 33,
    "feature extract": 33, "overfitting": 33, "backpropagat": 33,
    "convolutional": 33, "recurrent": 33, "gradient descent": 33,

    # ── Engineering (34–37) ──
    "mechanical": 34, "machine design": 34, "gear": 34, "bearing": 34,
    "turbine": 34, "hvac": 34, "pneumat": 34, "hydraul": 34,
    "manufactur": 34, "cnc": 34, "thermomech": 34,
    "combustion engine": 34, "internal combustion": 34, "engine design": 34,
    "jet engine": 34, "rocket engine": 34, "diesel engine": 34,
    "electric": 35, "circuit": 35, "semiconductor": 35, "transistor": 35,
    "voltage": 35, "current": 35, "capacitor": 35, "inductor": 35,
    "antenna": 35, "signal processing": 35, "power electronics": 35, "diode": 35,
    "material": 36, "composite": 36, "alloy": 36,
    "metallurg": 36, "crystal": 36, "biomaterial": 36, "corrosion": 36,
    "fatigue": 36, "fracture": 36, "deformation": 36,
    "transport": 37, "vehicle": 37, "traffic": 37, "logistics": 37,
    "railway": 37, "aviation": 37, "aerospace": 37, "maritime": 37,
    "automobile": 37, "highway": 37,

    # ── Nanotechnology (38–40) ──
    "nano": 38, "nanoparticle": 38, "nanotube": 38, "nanowire": 38,
    "nanocomposit": 38, "nanostructur": 38, "nanoscale": 38,
    "nanofabric": 38, "nanoelectron": 38, "nanomedicine": 38,
    "atom": 39, "atomic": 39, "isotope": 39,
    "electron": 39, "proton": 39, "neutron": 39, "valence": 39,
    "ionization": 39, "ionic": 39, "ionize": 39,
    "anionic": 39, "cationic": 39,
    "ionic bond": 39, "ionic compound": 39, "ionization energy": 39,
    "ion beam": 39, "ion implant": 39, "ionic radius": 39, "ion exchange": 39,
    "surface": 40, "thin film": 40, "coating": 40, "adsorption": 40,
    "wettab": 40, "tribolog": 40, "surface tension": 40,

    # ── Astronomy (41–44) ──
    "celestial": 41, "cosmos": 41, "universe": 41, "cosmolog": 41,
    "big bang": 41, "dark matter": 41, "dark energy": 41, "cosmic": 41,
    "stellar": 42, "supernova": 42, "neutron star": 42,
    "white dwarf": 42, "main sequence": 42, "nucleosynthesis": 42, "pulsar": 42,
    "star formation": 42, "binary star": 42, "star cluster": 42,
    "starlight": 42, "stars and galaxies": 42,
    "planet": 43, "exoplanet": 43, "terrestrial": 43, "jovian": 43,
    "moon": 43, "asteroid": 43, "comet": 43, "meteorit": 43, "mars": 43,
    "orbit": 44, "kepler": 44, "satellite": 44, "trajectory": 44,
    "eclips": 44, "perihelion": 44,

    # ── Earth Science (45–48) ──
    "geolog": 45, "tectonic": 45, "seismic": 45, "earthquake": 45,
    "volcani": 45, "sediment": 45, "stratigraphy": 45, "mineralog": 45,
    "lithosphere": 45, "mantle": 45, "geomorpholog": 45,
    "climate": 46, "global warming": 46, "greenhouse": 46, "ice age": 46,
    "paleoclim": 46, "meteorolog": 46, "weather": 46, "atmospher": 46,
    "precipitation": 46, "monsoon": 46, "el nino": 46,
    "ocean": 47, "marine": 47, "tidal": 47, "deep sea": 47,
    "coral": 47, "plankton": 47, "oceanograph": 47, "seafloor": 47,
    "water": 48, "hydro": 48, "aquifer": 48, "groundwater": 48,
    "watershed": 48, "desalinat": 48, "irrigation": 48, "freshwater": 48,
    "river": 48, "wetland": 48, "limnolog": 48,

    # ── Environmental Science (49–51) ──
    "ecosystem": 49, "food web": 49, "trophic": 49, "biome": 49,
    "habitat": 49, "ecological": 49, "keystone species": 49,
    "conservat": 50, "endangered": 50, "extinction": 50, "wildlife": 50,
    "protected area": 50, "biodiversity loss": 50, "rewild": 50,
    "sustainab": 50, "restoration ecology": 50,
    "nature": 51, "natural": 51, "wilderness": 51, "forest": 51,
    "vegetation": 51, "grassland": 51, "savanna": 51, "tundra": 51,

    # ── Psychology (52–54) ──
    "attachment": 52, "bonding": 52, "separation anxiety": 52,
    "secure base": 52, "caregiving": 52, "parent-child": 52,
    "trauma": 53, "ptsd": 53, "adverse childhood": 53, "abuse": 53,
    "grief": 53, "resilience": 53, "crisis": 53,
    "mental health": 54, "anxiety": 54, "depression": 54, "bipolar": 54,
    "schizophren": 54, "psychosis": 54, "well-being": 54, "mindful": 54,
    "coping": 54, "self-esteem": 54,

    # ── Philosophy (55–58) ──
    "ethic": 55, "moral philosophy": 55, "bioethic": 55, "deontolog": 55,
    "utilitarian": 55, "virtue ethic": 55, "normative": 55, "applied ethic": 55,
    "metaphysic": 56, "existence": 56,
    "reality": 56, "substance": 56, "causation": 56, "free will": 56,
    "determinism": 56, "possible world": 56,
    "epistemolog": 57, "knowledge": 57, "justification": 57, "belief": 57,
    "skepticism": 57, "empiricism": 57, "rationalism": 57, "a priori": 57,
    "ontology": 58, "categor": 58, "taxonomy": 58, "class hierarchy": 58,
    "mereolog": 58,

    # ── Religion (59–62) ──
    "spiritual": 59, "mysticism": 59, "meditation": 59, "contemplat": 59,
    "transcenden": 59, "prayer": 59, "soul": 59,
    "ritual": 60, "ceremony": 60, "liturgy": 60, "sacrament": 60,
    "pilgrimage": 60, "rite of passage": 60, "worship": 60,
    "sacred": 61, "holy": 61, "divine": 61, "scripture": 61,
    "revelation": 61, "prophet": 61, "temple": 61, "shrine": 61,
    "doctrine": 62, "dogma": 62, "creed": 62, "heresy": 62,
    "canon law": 62, "eschatology": 62, "soteriology": 62, "christolog": 62,

    # ── Linguistics (63–67) ──
    "language": 63, "bilingual": 63, "multilingual": 63, "dialect": 63,
    "creole": 63, "pidgin": 63, "lingua franca": 63, "sociolinguist": 63,
    "grammar": 64, "morpholog": 64, "inflect": 64, "conjugat": 64,
    "declension": 64, "part of speech": 64,
    "phonetic": 65, "phonolog": 65, "vowel": 65, "consonant": 65,
    "prosody": 65, "intonation": 65, "syllable": 65, "accent": 65,
    "syntax": 66, "parsing": 66, "phrase structure": 66, "dependency grammar": 66,
    "constituent": 66, "clause": 66, "sentence structure": 66,
    "semantic": 67, "meaning": 67, "pragmatic": 67, "discourse analysis": 67,
    "lexical": 67, "polysem": 67, "anaphora": 67,

    # ── Literature (68–70) ──
    "narrative": 68, "storytelling": 68, "plot": 68, "narrator": 68,
    "fiction": 68, "novel": 68, "short story": 68, "mythology": 68,
    "literary": 69, "literary criticism": 69, "genre": 69, "canon": 69,
    "modernism": 69, "postmodernism": 69, "realism": 69, "romanticism": 69,
    "comparative literature": 69, "intertextual": 69,
    "poetry": 70, "sonnet": 70, "haiku": 70,
    "stanza": 70, "rhyme": 70, "lyric": 70, "ballad": 70,
    "versification": 70, "free verse": 70, "blank verse": 70, "poetic verse": 70,

    # ── History (71–72) ──
    "history": 71, "historical": 71, "historiograph": 71, "medieval": 71,
    "ancient": 71, "colonial": 71, "postcolonial": 71, "revolution": 71,
    "empire": 71, "dynasty": 71, "civilization": 71,
    "warfare": 71, "wartime": 71, "post-war": 71, "world war": 71,
    "war strategy": 71, "civil war": 71, "cold war": 71,
    "archiv": 72, "manuscript": 72, "primary source": 72,
    "paleograph": 72, "epigraphy": 72,

    # ── Sociology (73–75) ──
    "society": 73, "social": 73, "inequality": 73,
    "stratification": 73, "urbanization": 73, "migration": 73, "demograph": 73,
    "community": 74, "civic": 74, "neighborhood": 74, "grassroot": 74,
    "volunte": 74, "collective action": 74, "solidarity": 74,
    "social network": 75, "social media": 75, "influenc": 75,
    "diffusion": 75, "homophily": 75,

    # ── Anthropology (76–78) ──
    "culture": 76, "cultural": 76, "intercultural": 76, "multicultural": 76,
    "cross-cultural": 76, "cultural identity": 76, "acculturat": 76,
    "tradition": 77, "indigenous": 77, "folklore": 77, "heritage": 77,
    "oral history": 77, "custom": 77,
    "kinship": 78, "family": 78, "marriage": 78, "clan": 78,
    "lineage": 78, "descent": 78, "household": 78,

    # ── Political Science (79–81) ──
    "governance": 79, "government": 79, "bureaucra": 79,
    "sovereignty": 79, "federalism": 79, "constitution": 79,
    "parliament": 79, "legislature": 79,
    "power": 80, "authority": 80, "legitimacy": 80, "hegemony": 80,
    "imperialism": 80, "totalitarian": 80, "authoritarian": 80,
    "policy": 81, "public policy": 81, "regulation": 81, "sanction": 81,
    "diplomacy": 81, "foreign policy": 81, "welfare": 81, "reform": 81,

    # ── Law (82–84) ──
    "legal": 82, "legislation": 82, "statute": 82,
    "jurisprudence": 82, "court": 82, "litigation": 82, "tort": 82,
    "contract": 82, "liability": 82, "compliance": 82,
    "justice": 83, "criminal justice": 83, "restorative justice": 83,
    "penal": 83, "sentencing": 83, "incarcerat": 83,
    "rights": 84, "human rights": 84, "civil rights": 84, "freedom": 84,
    "liberty": 84, "suffrage": 84, "emancipat": 84,

    # ── Economics (85–88) ──
    "market": 85, "supply and demand": 85, "competition": 85,
    "monopol": 85, "oligopol": 85, "trade": 85,
    "export": 85, "import": 85, "tariff": 85,
    "finance": 86, "investment": 86, "banking": 86, "stock": 86,
    "bond": 86, "portfolio": 86, "asset": 86, "venture capital": 86,
    "initial public offering": 86, "stock market": 86, "ipo market": 86,
    "labor": 87, "employment": 87, "wage": 87, "workforce": 87,
    "unemployment": 87, "union": 87, "collective bargain": 87,
    "human capital": 87, "gig economy": 87,
    "money": 88, "currency": 88, "inflation": 88, "deflation": 88,
    "monetary policy": 88, "central bank": 88, "interest rate": 88,
    "fiscal policy": 88, "gdp": 88, "taxation": 88,

    # ── Education (89–91) ──
    "pedagog": 89, "teaching": 89, "instruction": 89, "didactic": 89,
    "classroom": 89, "tutor": 89, "lecture": 89,
    "curriculum": 90, "syllabus": 90, "course design": 90,
    "accreditation": 90, "learning objective": 90, "competency": 90,
    "assessment": 91, "examination": 91, "grading": 91, "rubric": 91,
    "standardized test": 91, "formative assessment": 91, "summative": 91,
    "evaluation": 91,

    # ── Visual Arts (92–95) ──
    "visual": 92, "imagery": 92, "illustration": 92, "pictorial": 92,
    "engraving": 92, "mural": 92, "fresco": 92,
    "color": 93, "colour": 93, "pigment": 93, "palette": 93,
    "chromatic": 93, "hue": 93, "saturat": 93, "tint": 93,
    "shape": 94, "composition": 94, "proportion": 94,
    "symmetry": 94, "asymmetry": 94, "abstract": 94, "figurative": 94,
    "art form": 94, "formal composition": 94, "shape and form": 94,
    "visual form": 94, "geometric form": 94,
    "design": 95, "typograph": 95, "layout": 95, "user interface": 95,
    "industrial design": 95, "interior design": 95, "fashion": 95,

    # ── Music (96–99) ──
    "sound": 96, "audio": 96, "acoust": 96, "sonic": 96,
    "decibel": 96, "reverberat": 96,
    "harmony": 97, "chord": 97, "tonal": 97, "key signature": 97,
    "modulation": 97, "counterpoint": 97, "voice leading": 97,
    "rhythm": 98, "beat": 98, "tempo": 98, "syncopat": 98,
    "polyrhythm": 98, "groove": 98,
    "timbre": 99, "overtone": 99, "spectral": 99,
    "orchestrat": 99, "tonal color": 99,

    # ── Film (100–101) ──
    "cinema": 100, "film": 100, "movie": 100, "motion picture": 100,
    "screenplay": 100, "cinematograph": 100, "documentary": 100, "animation": 100,
    "montage": 101, "editing": 101, "mise en scene": 101, "shot composition": 101,

    # ── Performing Arts (102–103) ──
    "theatr": 102, "stage": 102, "playwr": 102,
    "acting": 102, "rehears": 102, "improvis": 102, "broadway": 102, "puppet": 102,
    "dance": 103, "choreograph": 103, "ballet": 103, "contemporary dance": 103,
    "folk dance": 103, "kinestheti": 103,

    # ── Culinary Arts (104–106) ──
    "taste": 104, "umami": 104, "bitter": 104, "sweet": 104,
    "salty": 104, "palate": 104, "gustatory": 104,
    "sourness": 104, "sour taste": 104, "sour flavor": 104,
    "flavor": 105, "aroma": 105, "spice": 105, "herbal": 105,
    "seasoning": 105, "infusion": 105, "marinade": 105,
    "culinary herb": 105, "fresh herbs": 105, "dried herbs": 105,
    "cooking": 106, "baking": 106, "roasting": 106, "frying": 106,
    "ferment": 106, "cuisine": 106, "gastronom": 106, "culinary": 106,
    "recipe": 106, "food science": 106,

    # ── Cross-cutting (107–127) ──
    "information": 107, "data flow": 107, "signal": 107, "encoding": 107,
    "channel": 107, "bandwidth": 107, "communication": 107,
    "system": 108, "feedback": 108, "control system": 108, "complex system": 108,
    "emergence": 108, "self-organiz": 108, "homeostasis": 108, "nonlinear": 108,
    "optimiz": 109, "heuristic": 109, "pareto": 109, "linear programming": 109,
    "metaheuristic": 109, "simulated annealing": 109, "genetic algorithm": 109,
    "constraint": 109, "objective function": 109,
    "pattern": 110, "fractal": 110, "tessellat": 110,
    "regularity": 110, "recurring": 110, "motif": 110,
    "structure": 111, "architectur": 111, "hierarchy": 111,
    "scaffold": 111, "lattice": 111, "topology": 111, "configurat": 111,
    "network": 112, "connectivity": 112, "centrality": 112,
    "cluster": 112, "small world": 112, "scale free": 112, "hub": 112,
    "space": 113, "spatial": 113, "dimension": 113, "coordinate": 113,
    "manifold": 113, "euclidean": 113, "geometric": 113,
    "performance": 114, "efficiency": 114, "throughput": 114, "latency": 114,
    "benchmark": 114, "productiv": 114,
    "measur": 115, "metric": 115, "calibrat": 115,
    "precision": 115, "accuracy": 115, "sensor": 115, "quantif": 115,
    "motion": 116, "kinematic": 116, "velocity": 116,
    "displacement": 116, "locomotion": 116,
    "cycle": 117, "periodic": 117, "circadian": 117,
    "seasonal": 117, "life cycle": 117, "feedback loop": 117,
    "behavior": 118, "behavioral": 118, "conduct": 118, "habit": 118,
    "response": 118, "stimulus": 118, "conditioning": 118, "instinct": 118,
    "emotion": 119, "affect": 119, "sentiment": 119, "empathy": 119,
    "mood": 119, "passion": 119, "fear": 119,
    "joyful": 119, "joy and": 119, "feelings of joy": 119,
    "concept": 120, "abstraction": 120, "idea": 120, "mental model": 120,
    "schema": 120, "prototype": 120,
    "theory": 121, "theoretical": 121, "hypothesis": 121,
    "paradigm": 121, "postulate": 121,
    "learning": 122, "training": 122, "skill acquis": 122,
    "cognitive develop": 122, "scaffolding": 122,
    "statistic": 123, "probability": 123, "bayesian": 123, "frequentist": 123,
    "correlation": 123, "variance": 123, "sampling": 123,
    "confidence interval": 123, "p-value": 123, "distribut": 123,
    "moral": 124, "morality": 124, "conscience": 124, "virtue": 124,
    "vices": 124, "moral vice": 124, "duty": 124, "obligation": 124,
    "discourse": 125, "rhetoric": 125, "argument": 125, "debate": 125,
    "persuasion": 125, "propaganda": 125, "public sphere": 125,
    "cognition": 126, "cognitive": 126, "perception": 126, "attention": 126,
    "memory": 126, "problem solving": 126,
    "decision making": 126, "executive function": 126,
    "mind": 127, "psyche": 127, "intellect": 127,
    "thought": 127, "introspect": 127, "metacognit": 127,
    "self-aware": 127, "mental state": 127, "mental process": 127,
    "mental imagery": 127, "mental representation": 127,
}


# ═══════════════════════════════════════════════════════════════════════
#  OPENALEX FIELD → SPHEREQL CATEGORY
#
#  FIELD_TO_CATEGORY: 1:1 mappings (field ID → single category)
#  FIELD_MULTI_MAP: 1:N mappings (field ID → keyword-routed category)
#    The "default" key is used when no keyword matches.
# ═══════════════════════════════════════════════════════════════════════

FIELD_TO_CATEGORY: dict[int, str] = {
    16: "chemistry",
    17: "computer_science",
    19: "earth_science",
    20: "economics",
    23: "environmental_science",
    25: "nanotechnology",
    26: "mathematics",
    27: "medicine",
    28: "neuroscience",
    32: "psychology",
}

FIELD_MULTI_MAP: dict[int, dict[str, str]] = {
    11: {  # Agricultural and Biological Sciences
        "default": "biology",
        "ecology": "environmental_science", "conservation": "environmental_science",
        "food": "culinary_arts", "nutrition": "culinary_arts",
    },
    12: {  # Arts and Humanities
        "default": "philosophy",
        # Linguistics first — "philosophy of language" descriptions otherwise
        # steal linguistics-coded concepts to philosophy.
        "linguistic": "linguistics", "phonet": "linguistics", "phonolog": "linguistics",
        "morpholog": "linguistics", "semiot": "linguistics", "syntax": "linguistics",
        "translat": "linguistics", "discourse analys": "linguistics",
        "history": "history", "histor": "history", "archeolog": "history",
        "archiv": "history", "classical": "history", "medieval": "history", "ancient": "history",
        "philoso": "philosophy", "ethic": "philosophy",
        "metaphys": "philosophy", "epistemol": "philosophy",
        "liter": "literature", "poetry": "literature", "fiction": "literature",
        "narrative": "literature", "novel": "literature", "drama": "literature", "writing": "literature",
        "language": "linguistics",
        "relig": "religion", "theolog": "religion", "spiritual": "religion",
        "sacred": "religion", "islam": "religion", "christian": "religion",
        "buddhis": "religion", "judais": "religion", "hindu": "religion",
        "music": "music", "composit": "music", "acoustic": "music",
        "melod": "music", "harmon": "music", "rhythm": "music", "song": "music",
        "visual": "visual_arts", "paint": "visual_arts", "sculpt": "visual_arts",
        "photograph": "visual_arts", "graphic": "visual_arts", "drawing": "visual_arts",
        "ceramic": "visual_arts", "gallery": "visual_arts", "art hist": "visual_arts", "aesthet": "visual_arts",
        "film": "film_studies", "cinema": "film_studies", "movie": "film_studies",
        "screen": "film_studies", "media stud": "film_studies", "television": "film_studies",
        "perform": "performing_arts", "theatr": "performing_arts", "dance": "performing_arts",
        "choreograph": "performing_arts", "opera": "performing_arts", "stage": "performing_arts",
        "architect": "architecture", "urban plan": "architecture", "building": "architecture", "landscape": "architecture",
    },
    13: {"default": "biology"},
    14: {"default": "economics"},
    15: {"default": "engineering", "nano": "nanotechnology", "polymer": "nanotechnology"},
    18: {"default": "data_science", "manage": "economics"},
    21: {"default": "engineering", "nuclear": "physics", "renewable": "environmental_science"},
    22: {
        "default": "engineering", "nano": "nanotechnology", "architect": "architecture",
        "civil": "architecture", "biomedic": "medicine", "computer": "computer_science", "software": "computer_science",
    },
    24: {"default": "biology", "immun": "medicine", "clinic": "medicine", "pathog": "medicine", "vaccin": "medicine"},
    29: {"default": "medicine"},
    30: {"default": "medicine"},
    31: {
        "default": "physics", "astro": "astronomy", "stellar": "astronomy", "galax": "astronomy",
        "planet": "astronomy", "cosmolog": "astronomy", "celestial": "astronomy",
        "solar system": "astronomy", "observat": "astronomy", "telescope": "astronomy",
    },
    33: {
        "default": "sociology",
        "anthropol": "anthropology", "ethnograph": "anthropology", "cultural stud": "anthropology",
        "politic": "political_science", "govern": "political_science", "democra": "political_science",
        "international rel": "political_science", "public admin": "political_science",
        "law": "law", "legal": "law", "jurispru": "law", "criminal": "law",
        "constitu": "law", "human right": "law", "justice": "law",
        "educa": "education", "pedagog": "education", "curricul": "education",
        "teach": "education", "school": "education", "higher ed": "education",
        "geograph": "earth_science", "urban": "architecture",
        "media": "film_studies", "gender": "sociology",
    },
    34: {"default": "biology"},
    35: {"default": "medicine"},
    36: {"default": "medicine"},
}


# ═══════════════════════════════════════════════════════════════════════
#  CATEGORY → PRIMARY AXES
#
#  Extracted from the hand-crafted 775-concept corpus. Each list gives
#  the axes most frequently activated in that category, ordered by
#  frequency. Used to seed features for concepts with sparse keyword
#  matches.
# ═══════════════════════════════════════════════════════════════════════

CATEGORY_PRIMARY_AXES: dict[str, list[int]] = {
    "physics":              [0, 7, 1, 2, 3, 121, 6, 9],
    "mathematics":          [7, 27, 111, 110, 8, 9, 11, 121],
    "biology":              [12, 13, 15, 14, 16, 51, 17, 49],
    "chemistry":            [16, 17, 111, 0, 18, 2, 36],
    "medicine":             [19, 12, 22, 108, 21, 16, 20],
    "neuroscience":         [12, 24, 126, 23, 112, 118, 127],
    "computer_science":     [26, 28, 27, 29, 7, 110, 108],
    "data_science":         [32, 26, 123, 110, 7, 33, 107, 109],
    "engineering":          [108, 111, 1, 34, 0, 36, 109],
    "nanotechnology":       [38, 36, 111, 16, 0, 40, 2],
    "astronomy":            [113, 0, 3, 42, 115, 41, 7],
    "earth_science":        [51, 45, 111, 46, 49, 1, 13],
    "environmental_science":[49, 51, 50, 108, 12, 16],
    "psychology":           [127, 118, 126, 110, 119, 24],
    "philosophy":           [127, 56, 55, 27, 118, 111, 124],
    "religion":             [56, 118, 63, 55, 62, 59, 61],
    "linguistics":          [63, 110, 111, 126, 118, 64, 67],
    "literature":           [63, 68, 69, 119, 110, 126],
    "history":              [71, 118, 108, 13, 63, 68],
    "sociology":            [118, 73, 108, 112, 110, 123],
    "anthropology":         [118, 76, 110, 13, 12, 108, 63],
    "political_science":    [79, 118, 108, 80, 81, 55],
    "law":                  [82, 27, 79, 55, 84, 85],
    "economics":            [85, 109, 118, 108, 7, 121, 123],
    "education":            [89, 126, 122, 118, 108, 90],
    "visual_arts":          [92, 110, 119, 36, 94, 111, 95],
    "music":                [96, 110, 119, 97, 114, 98, 7],
    "film_studies":         [100, 92, 68, 110, 119, 63],
    "performing_arts":      [114, 102, 116, 119, 103, 118],
    "culinary_arts":        [16, 106, 110, 114, 115, 105, 104],
    "architecture":         [111, 95, 94, 113, 92, 108, 36],
}


# ═══════════════════════════════════════════════════════════════════════
#  CONTENT OVERRIDES
#
#  Applied AFTER field-based routing. OpenAlex sometimes files topics in
#  the wrong field for our taxonomy (e.g., "Superconducting Materials"
#  in Field 27/29/35 defaults to medicine; "Geochemistry and Geologic
#  Mapping" in Field 17 lands in computer_science). Each rule overrides
#  the resolved category when the topic text strongly matches a keyword
#  and does NOT contain an excluded term.
#
#  Order matters: first matching rule wins. Use very specific keywords —
#  these overrides bypass the field signal and can introduce new bugs
#  if too broad.
# ═══════════════════════════════════════════════════════════════════════

CONTENT_OVERRIDES: list[tuple[str, str, str | None]] = [
    # geological concepts misrouted into engineering/CS via fields 17/22
    ("geolog", "earth_science", None),
    ("geochem", "earth_science", None),
    ("petrolog", "earth_science", None),
    ("stratigraph", "earth_science", None),
    # superconductivity / nanophysics defaulted to medicine via 27/29/35/36
    ("supercond", "physics", None),
    ("photocathod", "physics", None),
    ("microchannel plate", "physics", None),
    ("plasma physic", "physics", None),
    # legal content misrouted via field 27 (Medicine) defaults
    ("legal cases", "law", None),
    ("legal commentar", "law", None),
    ("constitutional law", "law", None),
    ("contract law", "law", None),
    # linguistics misrouted into philosophy/sociology; exclude computational
    # linguistics which legitimately belongs in CS/data_science.
    ("linguistics and discourse", "linguistics", "computational"),
    ("phonological", "linguistics", None),
    ("morphosyntactic", "linguistics", None),
]


# ═══════════════════════════════════════════════════════════════════════
#  DOMAIN AXIS RANGES (for bridge detection)
# ═══════════════════════════════════════════════════════════════════════

DOMAIN_AXIS_RANGES: list[range] = [
    range(0, 7), range(7, 12), range(12, 16), range(16, 19),
    range(19, 23), range(23, 26), range(26, 30), range(30, 34),
    range(34, 38), range(38, 41), range(41, 45), range(45, 49),
    range(49, 52), range(52, 55), range(55, 59), range(59, 63),
    range(63, 68), range(68, 71), range(71, 73), range(73, 76),
    range(76, 79), range(79, 82), range(82, 85), range(85, 89),
    range(89, 92), range(92, 96), range(96, 100), range(100, 102),
    range(102, 104), range(104, 107),
    # 107–127 are cross-cutting — not counted for bridge detection
]
