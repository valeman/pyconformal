"""Download the public datasets used in the real-data runs and store them as .npz (numeric features only)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_openml, fetch_california_housing, load_diabetes
out = sys.argv[1]
def save(name, X, y, src):
    X = pd.DataFrame(X).select_dtypes("number"); X = X.loc[:, X.notna().all()]
    y = np.asarray(y, float); ok = np.isfinite(y)
    np.savez_compressed(f"{out}/{name}.npz", X=X.values[ok].astype(float), y=y[ok], src=src)
    print(f"{name:12s} rows {ok.sum():6d}  features {X.shape[1]:3d}  {src}", flush=True)
d = load_diabetes(); save("diabetes", d.data, d.target, "sklearn load_diabetes")
for name, kw in [("ames", dict(name="ames_housing", version=1)), ("bike", dict(data_id=42712)), ("kin8nm", dict(data_id=189)),
                 ("ccpp", dict(data_id=44973)), ("superconduct", dict(data_id=44964)), ("abalone", dict(data_id=183)), ("california", None)]:
    try:
        if kw is None:
            d = fetch_california_housing(); save(name, d.data, d.target, "sklearn fetch_california_housing")
        else:
            d = fetch_openml(**kw, as_frame=True, parser="auto"); save(name, d.data, pd.to_numeric(d.target, errors="coerce"), f"OpenML {kw} '{d.details.get('name')}' v{d.details.get('version')}")
    except Exception as e:
        print(name, "FAILED", type(e).__name__, str(e)[:150], flush=True)
