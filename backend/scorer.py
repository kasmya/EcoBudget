from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from bs4 import BeautifulSoup
import requests

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")


def get_real_image_bytes(src, base_url=None):
    if not src:
        return 50_000, True

    url = src
    if src.startswith('//'):
        url = 'https:' + src
    elif base_url and not src.startswith('http'):
        if src.startswith('/'):
            url = base_url.rstrip('/') + src
        else:
            url = base_url.rstrip('/') + '/' + src

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
    content_root = soup.find(id='mw-content-text') or soup.find(id='bodyContent') or soup.find('body')

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
        resources.append({
            "type": "text",
            "content": infobox_text,
            "bytes": len(infobox_text.encode('utf-8'))
        })

    for tag in content_root.find_all(['p', 'li', 'h2', 'h3']):
        text = tag.get_text(strip=True)
        if text.startswith('↑') or text.startswith('^'):
            continue
        if text and len(text) > 20 and text not in seen_texts:
            seen_texts.add(text)
            resources.append({
                "type": "text",
                "content": text,
                "bytes": len(text.encode('utf-8'))
            })

    for img in content_root.find_all('img'):
        alt = img.get('alt', 'image')
        src = img.get('src', '')
        real_bytes, is_estimated = get_real_image_bytes(src, base_url)
        resources.append({
            "type": "image",
            "content": alt,
            "bytes": real_bytes,
            "bytes_estimated": is_estimated
        })

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


def rank_by_vpb(task_text, resources):
    text_resources = [r for r in resources if r['type'] == 'text']

    for r in text_resources:
        r['embed_utility'] = compute_utility(task_text, r['content'])

    candidates = sorted(text_resources, key=lambda r: r['embed_utility'], reverse=True)[:30]

    for r in candidates:
        score, answer = check_answerability(task_text, r['content'])
        r['utility'] = score
        r['extracted_answer'] = answer
        r['vpb'] = r['utility'] / max(r['bytes'], 1)

    ranked = sorted(candidates, key=lambda r: r['vpb'], reverse=True)
    return ranked


def find_best_answer_full_page(task_text, resources):
    text_resources = [r for r in resources if r['type'] == 'text']
    best_score = 0.0
    best_answer = None
    for r in text_resources:
        score, answer = check_answerability(task_text, r['content'])
        if score > best_score:
            best_score = score
            best_answer = answer
    return best_score, best_answer


def check_task_success(extracted_answer, ground_truth):
    if not extracted_answer:
        return False
    return (ground_truth.lower() in extracted_answer.lower()
            or extracted_answer.lower() in ground_truth.lower())
