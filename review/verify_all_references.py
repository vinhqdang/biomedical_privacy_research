import re
import urllib.request
import urllib.parse
import json
import time
import sys

def parse_bib_entries(bib_path):
    with open(bib_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Match @type{key, ...}
    pattern = r'@(\w+)\s*\{\s*([^,]+),([^@]*)\}'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    entries = []
    for m in matches:
        entry_type = m.group(1).lower()
        key = m.group(2).strip()
        body = m.group(3)
        
        fields = {}
        # Simple field parser
        field_matches = re.finditer(r'(\w+)\s*=\s*[\{"](.*?)[\}"]\s*(?:,|$)', body, re.DOTALL)
        for fm in field_matches:
            fname = fm.group(1).lower()
            fval = fm.group(2).strip().replace('\n', ' ')
            fval = re.sub(r'\s+', ' ', fval)
            fields[fname] = fval
            
        entries.append({
            'type': entry_type,
            'key': key,
            'title': fields.get('title', ''),
            'author': fields.get('author', ''),
            'year': fields.get('year', ''),
            'doi': fields.get('doi', ''),
            'journal': fields.get('journal', fields.get('booktitle', fields.get('institution', ''))),
            'url': fields.get('url', '')
        })
    return entries

def check_doi(doi):
    doi_clean = doi.strip()
    if doi_clean.startswith('http'):
        doi_clean = re.sub(r'https?://(?:dx\.)?doi\.org/', '', doi_clean)
        
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi_clean)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'ResearchAuditor/1.0 (mailto:vinh.dq4@buv.edu.vn)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                msg = data.get('message', {})
                titles = msg.get('title', [])
                actual_title = titles[0] if titles else ''
                authors = msg.get('author', [])
                author_names = [a.get('family', '') for a in authors if 'family' in a]
                return {
                    'status': 'FOUND',
                    'actual_title': actual_title,
                    'actual_authors': author_names,
                    'doi': doi_clean
                }
    except urllib.error.HTTPError as e:
        return {'status': f'HTTP_{e.code}', 'error': str(e)}
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

def search_crossref_title(title):
    query = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?rows=3&query.title={query}"
    req = urllib.request.Request(url, headers={'User-Agent': 'ResearchAuditor/1.0 (mailto:vinh.dq4@buv.edu.vn)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                items = data.get('message', {}).get('items', [])
                return items
    except Exception as e:
        return []

def title_similarity(t1, t2):
    w1 = set(re.findall(r'\b\w{4,}\b', t1.lower()))
    w2 = set(re.findall(r'\b\w{4,}\b', t2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / max(len(w1), len(w2))

def audit_all(bib_path):
    entries = parse_bib_entries(bib_path)
    print(f"Loaded {len(entries)} entries from {bib_path}.\n")
    
    results = []
    for i, entry in enumerate(entries, 1):
        key = entry['key']
        title = entry['title']
        doi = entry['doi']
        
        print(f"[{i}/{len(entries)}] Checking '{key}'...")
        
        status = 'UNKNOWN'
        details = {}
        
        if doi:
            doi_res = check_doi(doi)
            if doi_res.get('status') == 'FOUND':
                actual_title = doi_res.get('actual_title', '')
                sim = title_similarity(title, actual_title)
                if sim >= 0.4:
                    status = 'VERIFIED_OK'
                    details = {'sim': sim, 'actual_title': actual_title, 'doi': doi}
                else:
                    status = 'DOI_MISMATCH'
                    details = {'sim': sim, 'bib_title': title, 'actual_title': actual_title, 'doi': doi}
            else:
                status = f"DOI_FAIL ({doi_res.get('status')})"
                # Try search by title to see if paper exists under another DOI
                search_items = search_crossref_title(title)
                best_match = None
                best_sim = 0.0
                for item in search_items:
                    t_list = item.get('title', [])
                    if t_list:
                        s = title_similarity(title, t_list[0])
                        if s > best_sim:
                            best_sim = s
                            best_match = item
                if best_match and best_sim >= 0.5:
                    status = 'REAL_PAPER_WRONG_DOI'
                    details = {
                        'best_sim': best_sim,
                        'real_title': best_match.get('title', [''])[0],
                        'real_doi': best_match.get('DOI', ''),
                        'provided_doi': doi
                    }
                else:
                    status = 'HALLUCINATED_OR_UNINDEXED'
                    details = {'provided_doi': doi, 'best_sim': best_sim}
        else:
            # No DOI provided, search by title
            search_items = search_crossref_title(title)
            best_match = None
            best_sim = 0.0
            for item in search_items:
                t_list = item.get('title', [])
                if t_list:
                    s = title_similarity(title, t_list[0])
                    if s > best_sim:
                        best_sim = s
                        best_match = item
            if best_match and best_sim >= 0.5:
                status = 'VERIFIED_FOUND_ONLINE'
                details = {
                    'best_sim': best_sim,
                    'real_title': best_match.get('title', [''])[0],
                    'real_doi': best_match.get('DOI', '')
                }
            else:
                status = 'NO_DOI_NOT_FOUND'
                details = {'best_sim': best_sim}
                
        print(f"    --> Status: {status}")
        if status != 'VERIFIED_OK':
            print(f"        Details: {details}")
            
        results.append({
            'entry': entry,
            'status': status,
            'details': details
        })
        time.sleep(0.1) # Be nice to Crossref
        
    # Save report
    with open('review/reference_audit_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print("\n================== SUMMARY ==================")
    verified = [r for r in results if r['status'] in ('VERIFIED_OK', 'VERIFIED_FOUND_ONLINE')]
    problems = [r for r in results if r['status'] not in ('VERIFIED_OK', 'VERIFIED_FOUND_ONLINE')]
    print(f"Total entries: {len(results)}")
    print(f"Verified OK: {len(verified)}")
    print(f"Problems / Suspect: {len(problems)}")
    for p in problems:
        print(f" - [{p['status']}] {p['entry']['key']}: {p['entry']['title']}")

if __name__ == '__main__':
    audit_all('manuscript/references.bib')
