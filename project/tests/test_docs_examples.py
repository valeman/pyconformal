"""Run every Python example in docs/methods, so the pages cannot go stale."""
import pathlib
import re
import warnings

import pytest

PAGES = sorted((pathlib.Path(__file__).parent.parent / "docs" / "methods").glob("*.md"))
MODULES = {"sklearn": "sklearn", "numba": "numba", "lightgbm": "lightgbm"}


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_examples_run(page):
    text = page.read_text()
    needs = re.search(r"<!-- requires:(.*?)-->", text)
    for name in (needs.group(1).split() if needs else []):
        pytest.importorskip(MODULES[name])
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert blocks, "every method page needs an example"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile("\n".join(blocks), str(page), "exec"), {"__name__": "__docs__"})
