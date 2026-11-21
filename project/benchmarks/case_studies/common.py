import numpy as np, pandas as pd
import os
D = os.environ.get("PYCONFORMAL_DATA", "data") + "/"   # folder holding the Monash .tsf files

def read_tsf(name):
    out = {}
    with open(D + name, encoding="latin-1") as f:
        data = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@data"):
                data = True; continue
            if not data:
                continue
            parts = line.split(":")
            vals = np.array([float(v) if v != "?" else np.nan for v in parts[-1].split(",")])
            out[parts[0]] = (parts[1:-1], vals)
    return out

def sf_frame(y, start, freq, uid="s"):
    return pd.DataFrame({"unique_id": uid, "ds": pd.date_range(start, periods=len(y), freq=freq), "y": y})
