from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import re
import spacy

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents)


def extract_capitalized_terms(text):
    matches = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
    stopwords = {'What', 'When', 'Where', 'Who', 'Why', 'How', 'Which', 'Is', 'Does', 'Do'}
    return set(m.lower() for m in matches if m not in stopwords and len(m) > 2)


def entity_consistency_score(task_text, candidate_text):
    task_entities = extract_capitalized_terms(task_text)
    candidate_entities = extract_entities(candidate_text)

    if not task_entities:
        return 1.0
    if not candidate_entities:
        return 0.7

    overlap = task_entities & candidate_entities
    if overlap:
        return 1.0

    for te in task_entities:
        for ce in candidate_entities:
            if te in ce or ce in te:
                return 1.0

    return 0.3


def get_real_image_bytes(src, base_url=None):
    if not src:
        return 50_000, True
    url = src
    if src.startswith('//'):
        url = 'https:' + src
    elif base_url and not src.startswith('http'):
        url = base_url.rstrip('/') + (src if src.startswith('/') else '/' + src)
    headers = {'User-Agent': 'Mozilla/5.0 (EcoBudget Research Bot; contact: student-project)'}
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True, headers=headers)
        size = resp.headers.get('Content-Length')
        if size and int(size) > 500:
            return int(size), False
    except Exception:
        pass
    return 50_000, True


def parse_resources(html, base_url=None, debug=False):
    soup = BeautifulSoup(html, 'html.parser')
    content_root = (soup.find(id='mw-content-text')
                     or soup.find(id='bodyContent')
                     or soup.find('div', class_='mw-parser-output')
                     or soup.find('body')
                     or soup)

    for ref_section in content_root.find_all(['ol', 'div'], class_=['references', 'reflist', 'refbegin']):
        ref_section.decompose()
    for cite_tag in content_root.find_all('cite'):
        cite_tag.decompose()

    infobox = content_root.find('table', class_=lambda c: c and 'infobox' in c)
    infobox_text = ""
    if infobox:
        infobox_text = infobox.get_text(separator=' | ', strip=True)
        infobox.decompose()

    for junk in content_root.find_all(['script', 'style', 'nav', 'footer', 'sup', 'table']):
        junk.decompose()

    resources = []
    seen_texts = set()

    if infobox_text:
        resources.append({"type": "text", "content": infobox_text, "bytes": len(infobox_text.encode('utf-8'))})

    for tag in content_root.find_all(['p', 'li', 'h2', 'h3']):
        text = tag.get_text(separator=' ', strip=True)
        if text.startswith('↑') or text.startswith('^'):
            continue
        if text and len(text) > 20 and text not in seen_texts:
            seen_texts.add(text)
            resources.append({"type": "text", "content": text, "bytes": len(text.encode('utf-8'))})

    for img in content_root.find_all('img'):
        alt = img.get('alt', 'image')
        src = img.get('src', '')
        real_bytes, is_estimated = get_real_image_bytes(src, base_url)
        resources.append({"type": "image", "content": alt, "bytes": real_bytes, "bytes_estimated": is_estimated})

    return resources


def compute_utility(task_text, resource_text):
    emb_task = embed_model.encode(task_text, convert_to_tensor=True)
    emb_res = embed_model.encode(resource_text, convert_to_tensor=True)
    return float(util.cos_sim(emb_task, emb_res)[0][0])


def check_answerability(task_text, text_chunk):
    if len(text_chunk.strip()) < 10:
        return 0.0, None
    try:
        result = qa_model(question=task_text, context=text_chunk, handle_impossible_answer=True)
        if not result['answer'].strip():
            return 0.0, None
        return result['score'], result['answer']
    except Exception:
        return 0.0, None


def rank_by_vpb(task_text, resources, use_entity_filter=True):
    text_resources = [r for r in resources if r['type'] == 'text']
    for r in text_resources:
        r['embed_utility'] = compute_utility(task_text, r['content'])
    candidates = sorted(text_resources, key=lambda r: r['embed_utility'], reverse=True)[:30]

    for r in candidates:
        score, answer = check_answerability(task_text, r['content'])
        if use_entity_filter:
            consistency = entity_consistency_score(task_text, r['content'])
            r['entity_consistency'] = consistency
            score = score * consistency
        else:
            r['entity_consistency'] = 1.0
        r['utility'] = score
        r['extracted_answer'] = answer
        r['vpb'] = r['utility'] / max(r['bytes'], 1)

    return sorted(candidates, key=lambda r: r['vpb'], reverse=True)


def find_best_answer_full_page(task_text, resources, use_entity_filter=True):
    text_resources = [r for r in resources if r['type'] == 'text']
    best_score = 0.0
    best_answer = None
    for r in text_resources:
        score, answer = check_answerability(task_text, r['content'])
        if use_entity_filter:
            score = score * entity_consistency_score(task_text, r['content'])
        if score > best_score:
            best_score = score
            best_answer = answer
    return best_score, best_answer


def check_task_success(extracted_answer, ground_truth):
    if not extracted_answer:
        return False
    ea = extracted_answer.lower().strip()
    gt = ground_truth.lower().strip()
    if gt in ea or ea in gt:
        return True
    # word-order-agnostic fallback (handles "AD 80" vs "80 AD")
    ea_tokens = set(ea.replace(',', '').split())
    gt_tokens = set(gt.replace(',', '').split())
    if gt_tokens and gt_tokens.issubset(ea_tokens):
        return True
    return False


def check_task_success_v2(extracted_answer, selected_text, ground_truth):
    """
    Handles both formats:
    - ground_truth is a str -> substring match against extracted_answer (old behavior)
    - ground_truth is a dict with 'required_facts' -> checks presence in the
      full selected_text (not just the one QA-extracted span).
    Returns (success: bool, facts_matched: int, facts_total: int)
    """
    if isinstance(ground_truth, str):
        success = check_task_success(extracted_answer, ground_truth)
        return success, (1 if success else 0), 1

    required = ground_truth["required_facts"]
    threshold = ground_truth.get("match_threshold", 1.0)
    haystack = (selected_text or "").lower()

    matched = sum(1 for fact in required if fact.lower() in haystack)
    total = len(required)
    success = (matched / total) >= threshold if total else False
    return success, matched, total
