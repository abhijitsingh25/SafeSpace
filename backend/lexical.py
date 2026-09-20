"""BM25 lexical retrieval - the fallback when no embedding provider is configured.

Why this exists: a fresh clone has neither sentence-transformers (~2GB of torch) nor a
hosted embedding key, so the exam corpus was silently unused and every academic question
fell through to generic support. The corpus is 350 short passages of natural-language
Q&A, which is exactly the shape BM25 handles well, and it needs no model, no key, and no
network.

This is a fallback, not a replacement. Embeddings still win on paraphrase ("I'm terrified
of Monday" vs "exam anxiety"), so ``retrieval.retrieve()`` prefers them when available.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Deliberately small: with 350 documents, stripping too much leaves queries with nothing
# to match on. These are the terms that carry no signal in a corpus that is entirely
# about one topic.
_STOPWORD_TEXT = """
    a an and are as at be because been being but by can cant could did do does doing dont for
    from had has have having he her hers him his how i id ill im in into is it its ive
    just me more most my no nor not of on or our out over own s so some such t than that
    the their them then there these they this those to too very was we were what when
    where which while who why will with would you your youre"""

_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Standard BM25 parameters; k1 controls term-frequency saturation, b length normalisation.
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


class BM25Index:
    """Okapi BM25 over the corpus. Built in memory at startup; no artefacts on disk."""

    def __init__(self, documents: list[str]) -> None:
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.doc_count = len(documents)
        self.avg_length = sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0

        self.term_freqs: list[Counter] = [Counter(tokens) for tokens in self.doc_tokens]

        doc_freq: Counter = Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))

        # Standard BM25 idf, with the +1 that keeps it positive for terms appearing in
        # more than half the corpus - common here, since every document is about exams.
        self.idf = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def __len__(self) -> int:
        return self.doc_count

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (document index, score) for the k best matches, best first."""
        query_terms = tokenize(query)
        if not query_terms or not self.doc_count:
            return []

        scores = [0.0] * self.doc_count
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                length_norm = 1 - _B + _B * (self.doc_lengths[index] / self.avg_length)
                scores[index] += idf * (tf * (_K1 + 1)) / (tf + _K1 * length_norm)

        ranked = sorted(enumerate(scores), key=lambda pair: -pair[1])
        return [(index, score) for index, score in ranked[:k] if score > 0]
