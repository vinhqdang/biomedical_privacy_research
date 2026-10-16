import json

with open('review/reference_audit_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

print(f"Total entries: {len(report)}")
for r in report:
    k = r['entry']['key']
    t = r['entry']['title']
    doi = r['entry']['doi']
    st = r['status']
    det = r['details']
    print(f"\n[{st}] KEY: {k}")
    print(f"  Title: {t}")
    print(f"  DOI:   {doi}")
    if 'actual_title' in det:
        print(f"  Actual Title from DOI: {det['actual_title']}")
    if 'real_title' in det:
        print(f"  Crossref Found Match:  {det['real_title']} (DOI: {det.get('real_doi')}, Sim: {det.get('best_sim'):.2f})")
