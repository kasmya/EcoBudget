from scorer import parse_resources
from conditions import run_ecobudget

html = open('pages/page1.html').read()
res = parse_resources(html)
out = run_ecobudget('When was the Louvre museum established?', res)
print('answer:', out['answer'])
print('coverage:', out['coverage'], '| loads:', out['resources_loaded'], '| stop:', out['gate_reason'])
print('bytes:', out['bytes_used'])
print(out['coverage_detail'])
