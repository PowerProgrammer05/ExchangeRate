import requests
import pandas as pd

ctr = ["KOR", "US"]
code = {"INF":  ["FP.CPI.TOTL.ZG"], "IR": ["FR.INR.RINR", "FR.INR.LEND"], "GR": ["GC.TAX.TOTL.GD.ZS"], "TD": ["NE.EXP.GNFS.CD", "NE.IMP.GNFS.CD"]}
#Inflations, Interest Rate, Government Revenue, Public Debt, Trading Debt

#Global Params
params = {
    "format" : "json",
    "date": "1996:2024",
    "per_page": "1000"
}

index1 = {} #KOR
index2 = {} #US

#Interest Rates
for k in code.keys():
    for ir in code[k]:
        for ct in ctr:
            url = f"https://api.worldbank.org/v2/country/{ct}/indicator/{ir}"
            res = requests.get(url, params=params)
            data = res.json()

            if len(data) < 2 or not isinstance(data[1], list):
                print(f"No data for {ct} - {ir}")
                continue

            data = data[1]

            df = pd.DataFrame([{
                "YR": int(item["date"]),
                "VALUE": item["value"]
            } for item in data if item["value"] is not None])

            if df.empty or "YR" not in df.columns:
                print(f"No valid data for {ct} - {ir}")
                continue

            df = df.sort_values("YR", ascending=False).reset_index(drop=True)

            key = f"INT_{ir}"
            if ct == "KOR":
                index1[ir] = df
            else:
                index2[ir] = df
print(index1)
