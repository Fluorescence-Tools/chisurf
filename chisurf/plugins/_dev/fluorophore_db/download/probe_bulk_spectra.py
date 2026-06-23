
import urllib.request
import json

url = "https://www.fpbase.org/api/proteins/spectra/?format=json&limit=10"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (ChiSurf)'})
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read())
    results = data if isinstance(data, list) else data.get('results', [])
    for item in results:
        print(f"Slug: {item.get('slug')}, Spectra count: {len(item.get('spectra', []))}")
