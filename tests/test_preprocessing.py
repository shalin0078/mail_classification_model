from mail_classifier.preprocessing import combine_email_fields, normalize_email_text


def test_normalize_replaces_urls_and_emails():
    text = "Hello <b>User</b>, visit https://example.com/a and mail me@test.com"

    normalized = normalize_email_text(text)

    assert "urltoken" in normalized
    assert "emailtoken" in normalized
    assert "<b>" not in normalized
    assert "https://example.com/a" not in normalized


def test_combine_email_fields_keeps_context():
    combined = combine_email_fields("Body", subject="Subject", sender="a@example.com")

    assert "subject: Subject" in combined
    assert "sender: a@example.com" in combined
    assert "body: Body" in combined

