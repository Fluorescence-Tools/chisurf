
import urllib.request
import json

slug = "egfp"
url = f"https://www.fpbase.org/api/proteins/spectra/?format=json&protein__slug={slug}"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (ChiSurf)'})
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read())
    print(f"Data type: {type(data)}")
    if isinstance(data, list) and len(data) > 0:
        print(f"First item keys: {list(data[0].keys())}")
        if 'spectra' in data[0]:
            print(f"  Spectra count: {len(data[0]['spectra'])}")
            if len(data[0]['spectra']) > 0:
                print(f"    First spectrum keys: {list(data[0]['spectra'][0].keys())}")
