import pandas as pd, numpy as np
flt = [("item_id", ">=", "FOODS_3_000_CA_1"), ("item_id", "<=", "FOODS_3_999_CA_1")]
t = pd.read_parquet("target.parquet", filters=flt); t = t[t.item_id.str.endswith("_CA_1") & t.item_id.str.startswith("FOODS_3_")]
x = pd.read_parquet("temporal.parquet", filters=flt); x = x[x.item_id.isin(t.item_id.unique())]
Y = t.pivot(index="timestamp", columns="item_id", values="demand").sort_index()
P = x.pivot(index="timestamp", columns="item_id", values="sell_price").sort_index().reindex(Y.index)
snap = x.groupby("timestamp").snap_CA.first().reindex(Y.index)
Y.index = pd.to_datetime(Y.index); P.index = pd.to_datetime(P.index); snap.index = pd.to_datetime(snap.index)
print(Y.shape, Y.index[0], Y.index[-1], "price NaN share", float(P.isna().mean().mean()))
first_price = P.notna().idxmax(); first_sale = (Y > 0).idxmax()
for d in ("2011-01-29", "2011-06-30", "2011-12-31", "2012-06-30", "2012-12-31", "2013-06-30", "2013-12-31"):
    print(d, "items with a price by then:", int((first_price <= d).sum()), "| with a sale by then:", int((first_sale <= d).sum()))
end = pd.Timestamp("2013-12-31") + pd.Timedelta(days=600)
full = P.loc[:end].notna().all(); print("price on every day through the conformal period:", int(full.sum()))
for d in ("2012-01-01", "2012-07-01", "2013-01-01"):
    print("price on every day from", d, "to end of conformal period:", int(P.loc[d:end].notna().all().sum()))
Y.to_pickle("Y.pkl"); P.to_pickle("P.pkl"); snap.to_pickle("snap.pkl")
