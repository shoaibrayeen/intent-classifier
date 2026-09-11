from app.services.normalizer import normalize_text, tokenize


def test_normalize_collapses_whitespace_and_case():
    assert normalize_text("  Show   ME   Contracts \n") == "show me contracts"


def test_tokenize_keeps_digits_and_internal_apostrophes():
    assert tokenize("Microsoft's contracts, 2026!") == ["microsoft's", "contracts", "2026"]


def test_tokenize_folds_accents():
    assert tokenize("Résumé Ünicode") == ["resume", "unicode"]


def test_tokenize_empty_input():
    assert tokenize("") == []
    assert tokenize("   !!! ??? ") == []
