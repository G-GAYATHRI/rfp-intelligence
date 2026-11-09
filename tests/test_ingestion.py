from src.ingestion.classify import detect_doc_type, detect_addendum_number
from src.ingestion.cleaning import fix_text

def test_doc_type():
    assert detect_doc_type("Addendum 2 foo.pdf") == "addendum"
    assert detect_doc_type("Contract Affidavit.pdf") == "affidavit"
    assert detect_doc_type("Dell Laptop Specs.pdf") == "specs"
    assert detect_doc_type("page.html") == "bid_page"

def test_addendum_number():
    assert detect_addendum_number("Addendum 2 foo.pdf") == 2
    assert detect_addendum_number("RFP.pdf") is None

def test_hyphenation_fix():
    assert fix_text("compu-\nter") == "computer"

def test_addendum_number_with_no_label():
    assert detect_addendum_number("Addendum No. 2 - RFP.pdf") == 2
