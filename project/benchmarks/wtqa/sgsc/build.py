import pandas as pd, numpy as np
names = ["id", "dt", "ck", "ek", "gs", "cl", "gg", "ng", "ot"]
agg = None
for ch in pd.read_csv("window.csv", names=names, header=0, usecols=["id", "gs", "cl", "gg", "ng"], chunksize=4_000_000, dtype={"id": "int64", "gs": "float32", "cl": "float32", "gg": "float32", "ng": "float32"}):
    a = ch.groupby("id").agg(n=("gs", "size"), cl=("cl", "sum"), gg=("gg", "sum"), ng=("ng", "sum"))
    agg = a if agg is None else agg.add(a, fill_value=0)
print("customers in window:", len(agg)); plain = agg[(agg.cl == 0) & (agg.gg == 0) & (agg.ng == 0)]
print("plain load (no controlled load, no generation):", len(plain), "| rows per customer: max", int(plain.n.max()), "| with all rows:", int((plain.n == plain.n.max()).sum()), "| with >= 99.9%:", int((plain.n >= 0.999 * plain.n.max()).sum()))
ids = set(plain.index[plain.n >= 0.999 * plain.n.max()]); parts = []
for ch in pd.read_csv("window.csv", names=names, header=0, usecols=["id", "dt", "gs"], chunksize=4_000_000, dtype={"id": "int64", "gs": "float32"}):
    parts.append(ch[ch.id.isin(ids)])
d = pd.concat(parts); d["dt"] = pd.to_datetime(d.dt); Y = d.pivot_table(index="dt", columns="id", values="gs", aggfunc="first").sort_index()
print(Y.shape, Y.index[0], Y.index[-1], "missing share", float(Y.isna().mean().mean())); Y.to_pickle("Y.pkl")
