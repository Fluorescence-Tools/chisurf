
import argparse
import json
import urllib.request

parser = argparse.ArgumentParser(description="Count FPbase spectra records")
parser.add_argument("--db", help="Ignored compatibility option", default=None)
parser.parse_args()

url = "https://www.fpbase.org/api/proteins/spectra/?format=json"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (ChiSurf)'})
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read())
    print(f"Total items in list: {len(data)}")
