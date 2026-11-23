# TorchCP 1.2.1 audit: receipts to cite

Captured 23 November 2026, 16:17 to 16:19 UTC.

| Item | Value |
|---|---|
| Release | torchcp 1.2.1, uploaded to PyPI 14 October 2025 |
| Git tag and commit | `v1.2.1`, `3541cde18b2447cd54e7ca6a814eb0d60585c3f9` |
| `master` when checked | `627455f` (20 July 2026); all six faults present |
| Wheel `torchcp-1.2.1-py3-none-any.whl`, SHA-256 | `b78e8243f8647ebba66a420242de325f21c3260bde6b32d7f081863d5e74528e` |
| Source `torchcp-1.2.1.tar.gz`, SHA-256 | `0549bb26…93ad7d` (full value: PyPI file page) |
| Wheel against tag | 89 of 89 Python files byte-identical |
| Reference for fault 6 | `tiffanyding/class-conditional-conformal`, commit `3f7077854d1e88526e39d13c45111b507c34318a`, `utils/conformal_utils.py` lines 441 and 466 to 467 (`null_qhat='standard'`) |

| Fault | File, lines | SHA-256 of the file (wheel = tag) | Wayback snapshot | Screenshot |
|---|---|---|---|---|
| 1 CQR-R | `torchcp/regression/score/cqrr.py` 29, 39 | `88106f0a17dd2520d7f7f069e46de4ac6e3767f47ef866f9fd95caff292de588` | https://web.archive.org/web/20261123161805/https://raw.githubusercontent.com/ml-stat-Sustech/TorchCP/3541cde18b2447cd54e7ca6a814eb0d60585c3f9/torchcp/regression/score/cqrr.py | `torchcp_01` |
| 2 ACI clip | `torchcp/regression/predictor/aci.py` 197 | `6c43599204b814fec1325c37822059ba7817db91a64f5614f420a6a141fc7d0d` | https://web.archive.org/web/20261123161815/https://raw.githubusercontent.com/ml-stat-Sustech/TorchCP/3541cde18b2447cd54e7ca6a814eb0d60585c3f9/torchcp/regression/predictor/aci.py | `torchcp_02` |
| 3 AgACI | `torchcp/regression/predictor/agaci.py` 88, 104 to 113 | `55d787f61dae7b97d2ec865b612cdb36fbaf618077f7aaac74674f9a6e65eb03` | https://web.archive.org/web/20261123161824/https://raw.githubusercontent.com/ml-stat-Sustech/TorchCP/3541cde18b2447cd54e7ca6a814eb0d60585c3f9/torchcp/regression/predictor/agaci.py | `torchcp_03` |
| 4, 5 NorABS | `torchcp/regression/score/norabs.py` 224, 234, 286 to 289 | `4c92dbbcb8f3bc6cfdaa957a5f83851d4f811867c6a3ad32b88cd0ca453fef62` | none (archive rate limit) | `torchcp_05`, `torchcp_06` |
| 6 Clustered | `torchcp/classification/predictor/cluster.py` 323 to 324 | `976cf820579eb75c913bedbca3d2a4cbf26859e41cb9b6c1c83082c70eaecc0d` | none (archive rate limit) | `torchcp_04` |
| Margin | `torchcp/classification/score/margin.py` 43 | `53c4b20ca8a9cdc49326cef3990b2e2f4d9a62b609bc8cfb34d566ee94940d22` | none | `torchcp_07` |

Screenshots are our own stamped renders of the file at the commit (weakest receipt). Environment: Python 3.13, torch 2.15.0 (CPU), numpy 2.5.5, torch-geometric 2.8.1.
