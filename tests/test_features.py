from mail_classifier.features import EmailRiskFeatureExtractor


def test_risk_features_detect_url_urgency_and_money():
    extractor = EmailRiskFeatureExtractor()

    features = extractor.transform(["URGENT!!! Claim $500 at http://bad.example now"])

    assert features.shape == (1, 12)
    assert features[0][2] == 1
    assert features[0][4] == 3
    assert features[0][7] >= 1
    assert features[0][8] >= 1

