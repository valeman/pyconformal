# crepes 0.9.1 and PUNCC 0.9.3 audits: receipts to cite

Captured 23 November 2026, 15:14 to 15:22 UTC. Order of strength: Wayback Machine snapshot, PyPI file hash (a released file cannot be replaced), git commit hash, then our own stamped screenshots in `png/`.

## crepes

| Item | Value |
|---|---|
| Release | crepes 0.9.1, uploaded to PyPI 12 June 2026, 12:44 UTC |
| Git tag and commit | `v0.9.1`, `e930c22240756330799b1239b7dbe3235b54391a` |
| Wheel `crepes-0.9.1-py3-none-any.whl`, SHA-256 | `d1e0c4c212c72883f2be3d3b58242782040c5c40295453f561112d23deb29fd2` |
| Source `crepes-0.9.1.tar.gz`, SHA-256 | `14255bfeef3d3cb76fefa78f9cef1d05ecf106dfae6f2bf76096eb7f5524c099` |
| `base.py`, SHA-256 (wheel and commit identical) | `96d5697003f4239f171d58e5c22934df4dd72e9e2dc03c4b4839f87baa817740` |
| `extras.py`, SHA-256 (wheel, commit and `master` at `88a0dfa` identical) | `3f902f580a01c2a8fc08a6d4b20893e27cc823260241eccedfef5147a2ab8ef2` |
| `base.py` at the commit, snapshot | https://web.archive.org/web/20261123151722/https://raw.githubusercontent.com/henrikbostrom/crepes/e930c22240756330799b1239b7dbe3235b54391a/src/crepes/base.py |
| `extras.py` at the commit, snapshot | https://web.archive.org/web/20261123151949/https://raw.githubusercontent.com/henrikbostrom/crepes/e930c22240756330799b1239b7dbe3235b54391a/src/crepes/extras.py |

| Fault | File and lines | Other receipts | Screenshots |
|---|---|---|---|
| 1. Out-of-bag calibration ignores leave-one-out estimates | `base.py` line 4151 (`sigmas = de.apply(X)`); `extras.py` lines 732 to 736 and 667 to 687 | The notebook that uses the faulty call: https://web.archive.org/web/20261123151423/https://github.com/henrikbostrom/crepes/blob/e930c22240756330799b1239b7dbe3235b54391a/docs/crepes_nb_wrap.ipynb | `crepes_01`, `crepes_02`, `crepes_03` |
| 2. Unseen Mondrian category gives [0, 0] | `base.py` lines 898 and 916 to 949 | Issue #54: https://web.archive.org/web/20261123152052/https://github.com/henrikbostrom/crepes/issues/54 | `crepes_04` |
| 3. CRPS with sigmas | `base.py` lines 2841 to 2842 | Issue #40 (no snapshot made) | `crepes_05` |
| 4. `MondrianCategorizer` `NameError` | `extras.py` line 413 | | `crepes_06` |
| 5. Global random state not restored | `base.py` lines 2584 to 2597 | | `crepes_07` |
| Rank in floating point | `base.py` line 900 | | `crepes_08` |

## PUNCC

| Item | Value |
|---|---|
| Release | puncc 0.9.3, uploaded to PyPI 23 June 2026, 11:31 UTC |
| Git tag and commit | `v0.9.3`, `7a0db784b9f3418bb1068ae4de2ba1902ee2b194` |
| `main` also checked | commit `29a1219fdb40487a022990f20b99f859c7e1b938`, 20 August 2026: the four files below unchanged |
| Wheel `puncc-0.9.3-py3-none-any.whl`, SHA-256 | `26a177ce45fe9f7ade6938288b01ad1fc7a8d6df773560fb01bfdbdfbadd96de` |
| Source `puncc-0.9.3.tar.gz`, SHA-256 | `8652e316a793cfe5e713ba99f56e350fd67adf47721f0f29e77bf9c888601ec1` |
| `api/prediction_sets.py`, SHA-256 (wheel, commit and `main` identical) | `39847ffdca66635b1e319593a2fa1bf8a93b8824ae0945ea8196530b23bb8081` |
| `api/nonconformity_scores.py`, SHA-256 | `7277268cac4acc3956fc938e621f2026ae47f4ee007abba624d7cdafb4749d34` |
| `object_detection.py`, SHA-256 | `39afeeecfac8340acfab0b5a3595e75624b300d037783af382cc34db28aaf972` |
| `regression.py`, SHA-256 | `dda76283bed9235bce22129a82e5dadefb9207e3d614f323ee366f261870d468` |

| Fault | File and lines | Snapshots | Screenshots |
|---|---|---|---|
| 1. APS and RAPS set size capped by the number of test rows | `api/prediction_sets.py` lines 163 and 192. Same cap in v0.7.4 (21 December 2023) | https://web.archive.org/web/20261123151637/https://github.com/deel-ai/puncc/blob/7a0db784b9f3418bb1068ae4de2ba1902ee2b194/deel/puncc/api/prediction_sets.py | `puncc_01` |
| 2. Box-wise detection, wrong sign on the max sides | `api/nonconformity_scores.py` lines 229 and 403; `api/prediction_sets.py` lines 426 to 427 and 473 to 476; `object_detection.py` lines 130 to 134. In the code since commit `6e0a8f8` (12 January 2024), first released in v0.7.6 | Scores: https://web.archive.org/web/20261123152050/https://raw.githubusercontent.com/deel-ai/puncc/7a0db784b9f3418bb1068ae4de2ba1902ee2b194/deel/puncc/api/nonconformity_scores.py · Predictor: https://web.archive.org/web/20261123152133/https://raw.githubusercontent.com/deel-ai/puncc/7a0db784b9f3418bb1068ae4de2ba1902ee2b194/deel/puncc/object_detection.py · Theory page stating the same scores (line 532): https://web.archive.org/web/20261123152140/https://raw.githubusercontent.com/deel-ai/puncc/7a0db784b9f3418bb1068ae4de2ba1902ee2b194/docs/source/theory_overview.rst | `puncc_02` to `puncc_07` |
| 3. EnbPI crashes with `s=1` | `regression.py` lines 880 to 882 | | `puncc_08` |

## Limits

- The Internet Archive rate-limited further saves. No snapshot was made of the PyPI pages for either package, crepes issue #40, or PUNCC issues #100 and #101. The PyPI hashes above can be checked against pypi.org at any time.
- Screenshots 01 to 08 of each library are rendered from the file fetched at the commit URL shown in their stamp; the full files are in `source/` and the release files in `pypi/`.
- None of the faults marked "no report found" has been reported to the maintainers by us yet.
