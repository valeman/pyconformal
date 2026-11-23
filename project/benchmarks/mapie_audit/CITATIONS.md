# MAPIE 1.6.0 audit: receipts to cite

Use these when a finding is challenged. Order of strength: Wayback Machine snapshot (held by the Internet Archive, with its own timestamp), PyPI file hash (PyPI does not allow a released file to be replaced), git commit hash (changing a file changes the hash), then our own stamped screenshots.

## What was audited

| Item | Value |
|---|---|
| Release | MAPIE 1.6.0, uploaded to PyPI 19 October 2026, 09:08 UTC |
| Git tag and commit | `v1.6.0`, `f1cc3d07460b2d7198f557d40d60cd563c51822b` |
| Wheel `mapie-1.6.0-py3-none-any.whl`, SHA-256 | `6bb6b217f33f1301913d230efab68b8b6ec3c0dcdf1e7ad28ccce73435f71dfb` |
| Source `mapie-1.6.0.tar.gz`, SHA-256 | `471631634184c4cc22641b8a72572df5b5e281e237b0c00ec6c1c668b78767c2` |
| PyPI release page, snapshot | https://web.archive.org/web/20261122192543/https://pypi.org/project/MAPIE/1.6.0/ |
| `master` also checked | commit `e4c4f5f7daeaa7c8b09323852ad1997e506891a2`, 19 November 2026: faults 1, 3 and 4 unchanged |

The files inside the PyPI wheel are byte-identical to the files at the v1.6.0 commit for every file cited below (SHA-256 compared).

## Fault 1: ACI clips its working level

| Receipt | Value |
|---|---|
| File and line | `mapie/regression/time_series_regression.py`, line 330: `self.current_alpha[alpha_0] = np.clip(new_alpha_t, 0, 1)` |
| File SHA-256 (wheel and commit) | `c5b52741784168967697086166a53601eaf020aa99f14de5c1becb8eda7e8611` |
| Git blob | `5582d41905998f3ea38b8a3cae2c3d71c18bfbc0` |
| Source at the commit, snapshot | https://web.archive.org/web/20261122183823/https://github.com/scikit-learn-contrib/MAPIE/blob/f1cc3d07460b2d7198f557d40d60cd563c51822b/mapie/regression/time_series_regression.py |
| Docstring citing Gibbs and Candès (API page), snapshot | https://web.archive.org/web/20261122183927/https://mapie.readthedocs.io/en/v1.6.0/api/regression/ |
| Example claiming "the same results" as published ACI, snapshot | https://web.archive.org/web/20261122184000/https://mapie.readthedocs.io/en/v1.6.0/generated/regression/3-scientific-articles/plot_zaffran2022_comparison/ |
| Time series tutorial, snapshot | https://web.archive.org/web/20261122184111/https://mapie.readthedocs.io/en/v1.6.0/generated/regression/1-quickstart/plot_ts-tutorial/ |
| README "theoretical guarantees" line, snapshot | https://web.archive.org/web/20261122184208/https://github.com/scikit-learn-contrib/MAPIE/blob/f1cc3d07460b2d7198f557d40d60cd563c51822b/README.md |
| Screenshots | `png/20`, `21`, `24`, `25`, `02`, `03`, `04` |
| Paper | Gibbs and Candès, "Adaptive Conformal Inference Under Distribution Shift", NeurIPS 2021, Proposition 4.1 and Lemma 4.1. Correct treatment of levels outside (0, 1) is also set out in Szabadváry, arXiv:2409.15548, Definition 3; that paper does not mention MAPIE's clip |

## Fault 2: APS and RAPS with `include_last_label=False`

| Receipt | Value |
|---|---|
| Files | `mapie/conformity_scores/sets/aps.py` lines 356 to 359; `mapie/classification.py` lines 1069 to 1072 (docstring: "may result in a coverage higher than `1 - alpha`") |
| Source at the commit, snapshot | https://web.archive.org/web/20261122184201/https://github.com/scikit-learn-contrib/MAPIE/blob/f1cc3d07460b2d7198f557d40d60cd563c51822b/mapie/conformity_scores/sets/aps.py |
| API page, snapshot | https://web.archive.org/web/20261122184032/https://mapie.readthedocs.io/en/v1.6.0/api/classification/ |
| Issue #1021, snapshot | https://web.archive.org/web/20261122184306/https://github.com/scikit-learn-contrib/MAPIE/issues/1021 |
| Pull request #1024, snapshot | https://web.archive.org/web/20261122184359/https://github.com/scikit-learn-contrib/MAPIE/pull/1024 |
| Screenshots | `png/22`, `23`, `06` |

## Fault 3: default RCPS bound

| Receipt | Value |
|---|---|
| File and lines | `mapie/risk_control/methods.py`, lines 93 to 115 (two halves, `K_R_max += K_R`); lines 36 to 41 (WSR is the default) |
| File SHA-256 (wheel and commit) | `2843dc755c1e70957a230f9308f95acef218bb90e370d0e3aff2dac0409473c6` |
| Git blob | `10f7bb89f77fa37d003c64c3bf5a6e619d2e0901` |
| Source at the v1.6.0 commit, snapshot | https://web.archive.org/web/20261122192442/https://github.com/scikit-learn-contrib/MAPIE/blob/f1cc3d07460b2d7198f557d40d60cd563c51822b/mapie/risk_control/methods.py |
| Source in `master` on 19 November 2026, snapshot | https://web.archive.org/web/20261123132853/https://github.com/scikit-learn-contrib/MAPIE/blob/e4c4f5f7daeaa7c8b09323852ad1997e506891a2/mapie/risk_control/methods.py |
| Pull request #998 (fault explained, fix offered, closed unmerged), snapshot | https://web.archive.org/web/20261123131430/https://github.com/scikit-learn-contrib/MAPIE/pull/998 |
| Risk-control API page, snapshot | https://web.archive.org/web/20261122192518/https://mapie.readthedocs.io/en/v1.6.0/api/risk-control/ |
| Risk-control theory page, snapshot | https://web.archive.org/web/20261122192552/https://mapie.readthedocs.io/en/v1.6.0/content/risk-control/theory/ |
| Screenshots | `png/26`, `27` |
| Paper | Bates, Angelopoulos, Lei, Malik, Jordan, "Distribution-Free, Risk-Controlling Prediction Sets", 2021, the Waudby-Smith-Ramdas bound |

The Wayback snapshot of #998 does not render the conversation text, which GitHub loads by script. The text quoted in the audit was read from the live page on 23 November 2026, about 13:20 UTC. If it matters, save the page as a PDF from a browser as well.

## Fault 4: default Holm procedure

| Receipt | Value |
|---|---|
| File and lines | `mapie/risk_control/fwer_control.py`, lines 171 to 172 (`np.argmin` over p-values), lines 80 to 86 (loop stops at the first failure); `binary_classification.py` line 98 (Holm is the default) |
| File SHA-256 (wheel and commit) | `6b9b35ac253caf2e015d0cc1dc469d47fd2b501b98a2a874f40c541e8f0bee09` |
| Git blob | `45e97dded153687f03e3aab2f957e904d5dc2db3` |
| Source at the commit, snapshot | https://web.archive.org/web/20261122192453/https://github.com/scikit-learn-contrib/MAPIE/blob/f1cc3d07460b2d7198f557d40d60cd563c51822b/mapie/risk_control/fwer_control.py |
| Screenshots | `png/28`, `29`, `30` |

## Smaller finding: float32 inputs

| Receipt | Value |
|---|---|
| Issue #557 (December 2024, closed with a workaround), snapshot | https://web.archive.org/web/20261123131453/https://github.com/scikit-learn-contrib/MAPIE/issues/557 |

## Other snapshots

| Page | Snapshot |
|---|---|
| Documentation home, v1.6.0 | https://web.archive.org/web/20261122184221/https://mapie.readthedocs.io/en/v1.6.0/ |
| Regression API, latest | https://web.archive.org/web/20261122184140/https://mapie.readthedocs.io/en/latest/api/regression/ |

## Limits

- No Wayback snapshot was made of issues #1018, #1030 or #321.
- Our screenshots were taken in a cloud workspace; their times are that machine's UTC clock. They show what was there; the snapshots and hashes prove it to a third party.
- Nothing here has been reported to MAPIE by us yet. Fault 2 was reported by a user and fault 3 by pull request #998.
