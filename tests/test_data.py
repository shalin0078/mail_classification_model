from pathlib import Path

from mail_classifier.data import load_dataset


def test_load_dataset_without_optional_columns(tmp_path: Path):
    dataset = tmp_path / "emails.csv"
    dataset.write_text("text,label\nHello team,primary\nSale today,promotion\n", encoding="utf-8")

    x, y = load_dataset(dataset)

    assert len(x) == 2
    assert y.tolist() == ["primary", "promotion"]
    assert "body: Hello team" in x.iloc[0]

