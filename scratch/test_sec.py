import requests
import json

url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001050446.json"
headers = {
    "User-Agent": "MSTR-BTC Analytics Engine contact@mstr-btc-analytics.com"
}

try:
    print("Fetching facts from SEC...")
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Keys in facts json:", data.keys())
        us_gaap = data.get("facts", {}).get("us-gaap", {})
        print("Number of US-GAAP elements:", len(us_gaap))
        
        # Search for interesting keys related to debt or convertible notes
        keywords = ["debt", "convertible", "preferred", "note", "dividend"]
        matching_keys = []
        for key in us_gaap.keys():
            key_lower = key.lower()
            if any(kw in key_lower for kw in keywords):
                matching_keys.append(key)
                
        print("\nMatching keys (first 30):")
        for key in matching_keys[:30]:
            print(f"- {key}")
            
        # Let's save a summary of matching keys to a scratch file
        with open("scratch/sec_keys.json", "w") as f:
            json.dump(matching_keys, f, indent=2)
            
    else:
        print("Failed to fetch:", response.text[:200])
except Exception as e:
    print("Error:", e)
