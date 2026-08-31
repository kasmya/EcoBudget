from scorer import parse_resources, rank_by_vpb, check_answerability
from evidence import EvidenceCoverageTracker, embed_model
from sentence_transformers import util

html = open('pages/page1.html').read()
res = parse_resources(html)
task = 'When was the Louvre museum established?'
tracker = EvidenceCoverageTracker(task)
req = tracker.requirements[0]
print('REQUIREMENT:', req['text'], '| qtype:', req['qtype'], '| entities:', req['entities'])
print()

ranked = [r for r in rank_by_vpb(task, res, use_entity_filter=True) if r['type'] == 'text']
req_emb = embed_model.encode(req['text'], convert_to_tensor=True)

for r in ranked[:6]:
    pas_emb = embed_model.encode(r['content'], convert_to_tensor=True)
    sim = float(util.cos_sim(req_emb, pas_emb)[0][0])
    qa_score, answer = check_answerability(req['text'], r['content'])
    print(f'sim={sim:.2f} qa={qa_score:.2f} ans={answer!r}')
    print('   ', r['content'][:120])
    print()
