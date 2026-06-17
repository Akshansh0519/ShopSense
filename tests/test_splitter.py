import pandas as pd

from recommender.data.splitter import DataSplitter


def test_temporal_split_has_no_future_leakage():
    df = pd.DataFrame(
        {
            "t_dat": pd.date_range("2024-01-01", periods=10, freq="D"),
            "customer_id": [f"u{i}" for i in range(10)],
            "article_id": [f"i{i}" for i in range(10)],
        }
    )

    train, val, test = DataSplitter(0.6, 0.2, 0.2).split(df)

    assert train["t_dat"].max() < val["t_dat"].min()
    assert val["t_dat"].max() < test["t_dat"].min()
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
