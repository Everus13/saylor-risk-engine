import requests
import json

url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001050446.json"
headers = {
    "User-Agent": "MSTR-BTC Analytics Engine contact@mstr-btc-analytics.com"
}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    facts = data["facts"]["us-gaap"]
    
    keys_to_inspect = [
        "LongTermDebtNoncurrent",
        "LongTermDebtCurrent",
        "LongTermDebt",
        "PreferredStockValue",
        "ConvertibleLongTermNotesPayable",
        "ConvertibleDebt",
        "PreferredStockDividends",
        "PaymentsOfDividendsPreferredStock"
    ]
    
    print("--- SEC EDGAR Dynamically Parsed Facts ---")
    for key in keys_to_inspect:
        if key in facts:
            val_data = facts[key]
            units = val_data.get("units", {})
            for unit_name, reports in units.items():
                if reports:
                    latest = sorted(reports, key=lambda x: x.get('end', x.get('period', '')))[-1]
                    print(f"\nFact: {key} ({unit_name})")
                    print(f"  Latest Value: ${latest.get('val'):,}" if unit_name == 'USD' else f"  Latest Value: {latest.get('val')}")
                    print(f"  Date/Period: {latest.get('end', latest.get('period'))}")
                    print(f"  Form: {latest.get('form')}")
        else:
            print(f"\nFact: {key} -> NOT FOUND in MSTR XBRL facts")
else:
    print("Failed")
