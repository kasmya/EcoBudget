"""
Evidence Coverage Tracker — EcoBudget v2 core.
Knows WHAT facts a task requires, not just whether something was found.
"""

import re
import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# ── Answer form patterns by question type ─────────────────────────────────────

ANSWER_PATTERNS = {
    "when":     re.compile(r'\b(\d{1,4}\s*(AD|BC|BCE|CE)?|[A-Z][a-z]+ \d{4}|\d{4}s?)\b'),
    "where":    re.compile(r'\b[A-Z][a-zA-Z\s,]{2,40}\b'),
    "who":      re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
    "how_much": re.compile(r'\b\d+[\.,]?\d*\s*(KB|MB|GB|km|m|ft|kg|lb|USD|EUR|INR|'
                           r'₹|\$|%|GHz|Mbps|W|mAh|MP|hours?|days?)?\b', re.I),
    "what":     re.compile(r'.{8,}'),
    "compare":  re.compile(r'.{8,}'),
    "list":     re.compile(r'.{8,}'),
}

QUESTION_TYPES = {
    "when":     re.compile(r'\b(when|what year|what date|founded|established|built|completed|born|died)\b', re.I),
    "where":    re.compile(r'\b(where|what city|what country|what state|location of|capital of)\b', re.I),
    "who":      re.compile(r'\b(who|which person|which designer|which artist)\b', re.I),
    "how_much": re.compile(r'\b(how much|what price|what cost|how many|how tall|how long|how far)\b', re.I),
    "compare":  re.compile(r'\b(compare|versus|vs\.?|difference between|better|between)\b', re.I),
    "list":     re.compile(r'\b(list|enumerate|what are the|name the)\b', re.I),
}

ATTRIBUTE_KEYWORDS = {
    "price", "cost", "speed", "height", "weight", "power", "rate",
    "capacity", "battery", "resolution", "range", "latency", "display",
    "screen", "camera", "storage", "memory", "ram", "processor", "cpu",
    "gpu", "size", "dimension", "colour", "color", "warranty", "rating",
}


def classify_question(question):
    for qtype, pattern in QUESTION_TYPES.items():
        if pattern.search(question):
            return qtype
    return "what"


# ── Entity extraction ─────────────────────────────────────────────────────────

# Catches: "iPhone 15", "Galaxy S24", "5G NR", "Wi-Fi 6E", "MacBook Pro"
PRODUCT_RE = re.compile(
    r'\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z0-9][a-zA-Z0-9]*)*'
    r'(?:\s+(?:\d+[A-Za-z]*|Pro|Max|Ultra|Plus|Mini|Air|Lite|SE|XR|XS))?)\b'
)


def extract_entities(text):
    """Extract named entities using spaCy + regex fallback for product names."""
    doc = nlp(text)
    entities = set()

    # spaCy entities
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "PRODUCT",
                          "WORK_OF_ART", "EVENT", "FAC"):
            entities.add(ent.text.strip())

    # Multi-word noun phrases
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) >= 2:
            entities.add(chunk.text.strip())

    # Regex: capitalized sequences that look like product names
    for m in PRODUCT_RE.finditer(text):
        candidate = m.group(1).strip()
        # Must have at least 2 chars, not be a pure stopword
        if len(candidate) > 2 and candidate not in {
            "The", "This", "That", "These", "Those", "What", "When",
            "Where", "Who", "How", "Compare", "And", "On", "In", "Of"
        }:
            entities.add(candidate)

    return entities


def extract_subjects(question):
    """
    Extract the things being compared/described.
    Uses both spaCy NER and the product regex, deduplicates,
    and filters out attribute words and stopwords.
    """
    stopwords = {
        "compare", "comparison", "difference", "versus", "vs", "between",
        "on", "by", "and", "or", "the", "a", "an", "in", "of", "for",
        "with", "what", "when", "where", "who", "how", "which",
        "price", "cost", "battery", "camera", "resolution", "speed",
        "height", "weight", "life", "display", "screen", "storage",
    }

    candidates = extract_entities(question)
    subjects = []
    for c in candidates:
        words = c.lower().split()
        if not all(w in stopwords for w in words):
            subjects.append(c)

    # Deduplicate: remove candidates that are subsets of longer ones
    subjects.sort(key=len, reverse=True)
    deduped = []
    for s in subjects:
        if not any(s.lower() in longer.lower() and s != longer for longer in deduped):
            deduped.append(s)
    return deduped


def extract_attributes(question):
    """Extract the attributes to compare (words after 'on', 'by', 'for', etc.)"""
    attr_match = re.search(r'\b(on|by|regarding|for|including)\b\s+(.+)', question, re.I)
    if not attr_match:
        return []
    attr_text = attr_match.group(2).rstrip('?.')
    raw = re.split(r',\s*|\s+and\s+|\s+&\s+', attr_text)
    return [a.strip() for a in raw if a.strip()]


# ── Requirement extraction ────────────────────────────────────────────────────

def extract_requirements(question):
    """
    Decompose a question into specific information requirements.
    Returns a list of requirement dicts.
    """
    qtype = classify_question(question)
    requirements = []

    if qtype == "compare":
        subjects = extract_subjects(question)
        attributes = extract_attributes(question)

        if subjects and attributes:
            for subj in subjects:
                for attr in attributes:
                    attr_qtype = "how_much" if any(
                        kw in attr.lower() for kw in ATTRIBUTE_KEYWORDS
                    ) else "what"
                    requirements.append({
                        "text": f"{subj} {attr}",
                        "qtype": attr_qtype,
                        "entities": {subj, attr},
                        "satisfied": False,
                        "answer": None,
                    })

    if not requirements:
        # Simple question: one requirement
        entities = extract_entities(question)
        requirements.append({
            "text": question.rstrip("?").strip(),
            "qtype": qtype,
            "entities": entities,
            "satisfied": False,
            "answer": None,
        })

    return requirements


# ── Passage evaluation ────────────────────────────────────────────────────────

def passage_satisfies_requirement(passage_text, requirement, qa_model_fn,
                                   similarity_threshold=0.40,
                                   qa_confidence_threshold=0.30):
    """
    Returns (satisfies: bool, answer: str, confidence: float).
    All four checks must pass for a passage to satisfy a requirement.
    """
    # 1. Semantic similarity to THIS specific requirement
    req_emb = embed_model.encode(requirement["text"], convert_to_tensor=True)
    pas_emb = embed_model.encode(passage_text, convert_to_tensor=True)
    sim = float(util.cos_sim(req_emb, pas_emb)[0][0])
    if sim < similarity_threshold:
        return False, None, 0.0

    # 2. Subject entity must appear in passage
    passage_lower = passage_text.lower()
    subject_found = False
    for entity in requirement["entities"]:
        # Check individual words of the entity too (handles "iPhone 15" → "iphone")
        entity_words = [w for w in entity.lower().split() if len(w) > 2]
        if entity.lower() in passage_lower or any(w in passage_lower for w in entity_words):
            subject_found = True
            break
    if not subject_found:
        return False, None, 0.0

    # 3. QA model extracts a confident answer
    qa_score, answer = qa_model_fn(requirement["text"], passage_text)
    if qa_score < qa_confidence_threshold or not answer:
        return False, None, 0.0

    # 4. Answer matches expected form for this requirement type
    pattern = ANSWER_PATTERNS.get(requirement["qtype"], ANSWER_PATTERNS["what"])
    if not pattern.search(answer):
        return False, None, 0.0

    return True, answer, qa_score


# ── Coverage tracker ──────────────────────────────────────────────────────────

class EvidenceCoverageTracker:
    def __init__(self, question):
        self.question = question
        self.requirements = extract_requirements(question)
        self.bytes_used = 0
        self.passages_checked = 0

    def n_satisfied(self):
        return sum(1 for r in self.requirements if r["satisfied"])

    def n_total(self):
        return len(self.requirements)

    def coverage(self):
        if not self.requirements:
            return 1.0
        return self.n_satisfied() / self.n_total()

    def is_sufficient(self, threshold=0.8):
        return self.coverage() >= threshold

    def check_passage(self, passage_text, passage_bytes, qa_model_fn):
        self.bytes_used += passage_bytes
        self.passages_checked += 1
        any_satisfied = False
        for req in self.requirements:
            if req["satisfied"]:
                continue
            ok, answer, confidence = passage_satisfies_requirement(
                passage_text, req, qa_model_fn
            )
            if ok:
                req["satisfied"] = True
                req["answer"] = answer
                req["confidence"] = confidence
                any_satisfied = True
        return any_satisfied

    def best_answer(self):
        satisfied = [r for r in self.requirements if r["satisfied"]]
        if not satisfied:
            return None
        if len(satisfied) == 1:
            return satisfied[0]["answer"]
        parts = [f"{r['text']}: {r['answer']}" for r in satisfied]
        return " | ".join(parts)

    def status_report(self):
        lines = []
        for r in self.requirements:
            mark = "✓" if r["satisfied"] else "✗"
            ans = f" → {r['answer']}" if r["satisfied"] else ""
            lines.append(f"  {mark} {r['text']}{ans}")
        return "\n".join(lines)
