import spacy
from nltk.tokenize import sent_tokenize
from collections import defaultdict

# ✅ Load the large English model with word vectors
nlp = spacy.load("en_core_web_lg")

# ✅ Print debug confirmation at runtime
print(f"[INFO] Loaded SpaCy model: {nlp.meta['name']}")  # Must say "en_core_web_lg"

def build_similarity_chains(sentences, top_k=5):
    """
    Compute SpaCy-based cosine similarities between all sentences,
    then for each sentence i select its top_k most similar peers.

    Args:
        sentences (List[str]): List of document sentences.
        top_k (int): Number of top similar sentences to consider per sentence.

    Returns:
        Dict[int, List[int]]: Mapping from sentence index to list of similar sentence indices.
    """
    docs = [nlp(s) for s in sentences]
    n = len(docs)

    # ✅ Check for any empty vectors (optional)
    if any(not doc.has_vector for doc in docs):
        print("[WARNING] Some sentences have no vectors! This may affect similarity.")

    # Build similarity matrix
    sim_matrix = [[docs[i].similarity(docs[j]) for j in range(n)] for i in range(n)]

    chains = defaultdict(list)
    for i in range(n):
        sims = list(enumerate(sim_matrix[i]))
        sims.sort(key=lambda x: x[1], reverse=True)
        for idx, score in sims[1: top_k+1]:
            chains[i].append(idx)
    return chains

def select_chain_sentences(chains, per_chain=2):
    """
    From the similarity chains, pick up to per_chain sentences for each seed.

    Args:
        chains (Dict[int,List[int]]): Mapping from seed index to similar indices.
        per_chain (int): How many top sentences per seed to select.

    Returns:
        Set[int]: Set of selected sentence indices.
    """
    selected = set()
    ranked = sorted(chains.items(), key=lambda x: len(x[1]), reverse=True)
    for seed_idx, neighbors in ranked:
        for neighbor in neighbors[:per_chain]:
            selected.add(neighbor)
    return selected
