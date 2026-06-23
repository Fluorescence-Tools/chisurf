
import urllib.request
import json

url = "https://www.fpbase.org/api/proteins/?format=json&limit=5"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (ChiSurf)'})
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read())
    results = data if isinstance(data, list) else data.get('results', [])
    for p in results[:5]:
        print(f"Name: {p.get('name')}, Slug: {p.get('slug')}")
        print(f"Keys: {list(p.keys())}")
        if 'states' in p and p['states']:
             print(f"  State keys: {list(p['states'][0].keys())}")
        print("-" * 20)
