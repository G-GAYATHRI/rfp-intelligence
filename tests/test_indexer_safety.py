import pytest, sys, types
from src.search import indexer


def test_indexer_does_not_replace_index_when_parser_returns_no_pages(monkeypatch, tmp_path):
    monkeypatch.setattr(indexer, "parse_bid_folder", lambda folder: [])
    called = {"replace": False}
    fake_store = types.SimpleNamespace(replace_bid=lambda *a: called.__setitem__("replace", True))
    monkeypatch.setitem(sys.modules, "src.search.store", fake_store)
    with pytest.raises(RuntimeError, match="No readable pages"):
        indexer.index_bid_folder(tmp_path)
    assert called["replace"] is False
