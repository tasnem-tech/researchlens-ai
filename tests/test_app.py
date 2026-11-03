
import pandas as pd

def test_missing_values():
    df = pd.DataFrame({
        "Age": [20, 25, None],
        "Score": [80, 90, 100]
    })
    assert df.isna().sum()["Age"] == 1

def test_duplicate_rows():
    df = pd.DataFrame({
        "Age": [20, 20, 30],
        "Score": [80, 80, 90]
    })
    assert df.duplicated().sum() == 1

def test_csv_export():
    df = pd.DataFrame({
        "Name": ["A", "B"],
        "Score": [85, 90]
    })
    csv_data = df.to_csv(index=False)
    assert "Name,Score" in csv_data
    assert "A,85" in csv_data
