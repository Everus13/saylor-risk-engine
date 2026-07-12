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
    
    # 1. Let's look at DebtInstrumentFaceAmount
    face_amount_data = facts.get("DebtInstrumentFaceAmount", {})
    print("DebtInstrumentFaceAmount units:", face_amount_data.get("units", {}).keys())
    
    # Typically USD
    usd_units = face_amount_data.get("units", {}).get("USD", [])
    print(f"Number of reports in DebtInstrumentFaceAmount: {len(usd_units)}")
    
    # Let's inspect unique dimensions (which contain notes descriptions)
    note_members = {}
    for item in usd_units:
        dims = item.get("dims", {})
        # Let's find members related to DebtInstrumentAxis
        for axis, member in dims.items():
            if "DebtInstrumentAxis" in axis or "LongtermDebtTypeAxis" in axis:
                note_members[member] = item
                
    print("\n--- Found Debt Instrument Members ---")
    for member, sample in list(note_members.items())[:20]:
        print(f"Member: {member}")
        print(f"  Form: {sample.get('form')}, Period: {sample.get('fy')}-{sample.get('fp')}, End Date: {sample.get('end')}")
        print(f"  Val: ${sample.get('val'):,}")
        
    # 2. Let's check DebtInstrumentInterestRateStatedPercentage
    interest_data = facts.get("DebtInstrumentInterestRateStatedPercentage", {})
    if interest_data:
        int_units = interest_data.get("units", {}).get("pure", [])
        print(f"\nNumber of reports in stated interest rates: {len(int_units)}")
        int_members = {}
        for item in int_units:
            dims = item.get("dims", {})
            for axis, member in dims.items():
                if "DebtInstrumentAxis" in axis:
                    int_members[member] = item.get("val")
        print("\nStated interest rates for members:")
        for m, rate in int_members.items():
            print(f"  {m}: {rate * 100}%" if rate is not None else f"  {m}: None")

    # 3. Let's inspect the general LongTermDebtNoncurrent value
    ltd_noncurrent = facts.get("LongTermDebtNoncurrent", {})
    if ltd_noncurrent:
        ltd_units = ltd_noncurrent.get("units", {}).get("USD", [])
        if ltd_units:
            latest = sorted(ltd_units, key=lambda x: x.get('end', ''))[-1]
            print(f"\nLatest LongTermDebtNoncurrent: ${latest.get('val'):,} as of {latest.get('end')}")

else:
    print("Failed")
