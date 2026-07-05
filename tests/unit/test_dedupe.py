from app.ingestion.dedupe import content_hash, normalize_text


def test_identical_text_same_hash():
    assert content_hash("Hello, World!") == content_hash("Hello, World!")


def test_case_and_punctuation_insensitive():
    assert content_hash("Hello, World!") == content_hash("hello world")


def test_whitespace_insensitive():
    assert content_hash("Hello   World") == content_hash("Hello World")


def test_different_text_different_hash():
    assert content_hash("Employee handbook section 3") != content_hash("Employee handbook section 4")


def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("Sick Days: Up to 10/year!") == "sick days up to 10year"
