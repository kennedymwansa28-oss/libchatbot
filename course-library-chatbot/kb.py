"""
Read-only knowledge base: loads the pre-built corpus and searches it with BM25.
Nothing here writes to the corpus; the chatbot can neither add nor remove documents.
"""
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent
CORPUS_FILE = ROOT / "data" / "corpus.json.gz"
SUGGESTIONS_FILE = ROOT / "data" / "suggestions.json"

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers him his
how i if in into is it its itself just me more most my no nor not of off on once only or other our out over own same
she should so some such than that the their theirs them then there these they this those through to too under until
up very was we were what when where which while who whom why will with would you your yours tell explain describe
give me please define list say used use using work works
""".split())

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#\-]*")


def stem(word: str) -> str:
    """Very light stemmer so 'attacks'/'attacker'/'attacking' match."""
    for suffix in ("ations", "ation", "ingly", "ings", "ing", "ers", "er", "ies", "ied", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
    return word


def tokenize(text: str):
    words = _TOKEN.findall(text.lower())
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


class KnowledgeBase:
    def __init__(self):
        with gzip.open(CORPUS_FILE, "rt", encoding="utf-8") as f:
            data = json.load(f)
        self.documents = {d["id"]: d for d in data["documents"]}
        self.chunks = data["chunks"]
        tokenized = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized, k1=1.4, b=0.75)
        self.df = Counter(t for toks in tokenized for t in set(toks))   # chunks containing each term
        # document frequency of each term, used for the relevance check
        self.idf = self.bm25.idf
        with open(SUGGESTIONS_FILE, encoding="utf-8") as f:
            self.suggestions = json.load(f)

    # ---------------------------------------------------------------- search
    def search(self, question: str, k: int = 6, per_doc: int = 3):
        """Return (passages, in_scope). Each passage is a dict with doc title, page, text, score."""
        q_tokens = tokenize(question)
        if not q_tokens:
            return [], False
        scores = self.bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:60]

        picked, per_doc_count = [], Counter()
        for i in order:
            c = self.chunks[i]
            if scores[i] <= 0:
                break
            if per_doc_count[c["doc"]] >= per_doc:
                continue
            per_doc_count[c["doc"]] += 1
            picked.append({
                "doc": c["doc"], "title": self.documents[c["doc"]]["title"],
                "page": c["page"], "page_end": c["page_end"],
                "text": c["text"], "score": float(scores[i]),
            })
            if len(picked) == k:
                break

        return picked, self._in_scope(q_tokens, picked)

    def coverage(self, q_tokens, passage_text) -> float:
        """Share of the question's (IDF-weighted) terms found in a single passage."""
        distinct = set(q_tokens)
        max_idf = max(self.idf.values())
        weight = {t: max(self.idf.get(t, max_idf), 0.1) for t in distinct}   # unknown words count fully
        tokens = set(tokenize(passage_text))
        return sum(w for t, w in weight.items() if t in tokens) / sum(weight.values())

    def _in_scope(self, q_tokens, picked) -> bool:
        """Decide whether the question is actually covered by the documents."""
        if not picked:
            return False
        distinct = set(q_tokens)
        # A one-word question must name something the documents really discuss, not a passing mention.
        if len(distinct) == 1 and self.df.get(next(iter(distinct)), 0) < 8:
            return False
        covs = [self.coverage(q_tokens, p["text"]) for p in picked[:5]]
        self.last_coverage = covs
        return max(covs) >= 0.7 and sum(c >= 0.5 for c in covs) >= 2

    # ------------------------------------------------------- suggested questions
    @staticmethod
    def _norm(q: str) -> str:
        return re.sub(r"\W+", " ", q.lower()).strip()

    def topics(self):
        return [t["name"] for t in self.suggestions["topics"]]

    def topic_questions(self, topic_name: str):
        for t in self.suggestions["topics"]:
            if t["name"] == topic_name:
                return list(t["questions"])
        return []

    def starter_questions(self, n=6, exclude=()):
        """One not-yet-asked question from each of the first n topics (a spread across the library)."""
        seen = {self._norm(q) for q in exclude}
        out = []
        for t in self.suggestions["topics"]:
            for q in t["questions"]:
                if self._norm(q) not in seen:
                    out.append(q)
                    break
            if len(out) == n:
                break
        return out

    def follow_ups(self, question, passages, asked=(), n=3):
        """Suggest questions related to what was just answered (same books / overlapping terms)."""
        seen = {self._norm(q) for q in asked} | {self._norm(question)}
        source_docs = {p["doc"] for p in passages[:4]}
        context = set(tokenize(question))
        for p in passages[:2]:
            context |= set(tokenize(p["text"]))
        scored = []
        for t in self.suggestions["topics"]:
            same_books = bool(set(t["docs"]) & source_docs)
            for q in t["questions"]:
                if self._norm(q) in seen:
                    continue
                q_tokens = set(tokenize(q))
                total = sum(self.idf.get(x, 0.1) for x in q_tokens) or 1.0
                overlap = sum(self.idf.get(x, 0.1) for x in q_tokens if x in context) / total
                scored.append((3.0 * same_books + 2.0 * overlap, t["name"], q))
        scored.sort(key=lambda x: x[0], reverse=True)
        out, per_topic = [], Counter()
        for _, topic, q in scored:
            if per_topic[topic] >= 2:
                continue
            per_topic[topic] += 1
            out.append(q)
            if len(out) == n:
                break
        return out
