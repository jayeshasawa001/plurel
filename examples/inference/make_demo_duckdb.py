"""Build a tiny sample DuckDB + labeled task tables that match config.py's demo
defaults, so you can run the 1 -> 2 -> 3 walkthrough without your own database.

    pixi run python examples/inference/make_demo_duckdb.py
    pixi run python examples/inference/1_data_prep.py
    pixi run python examples/inference/2_task_prep.py
    pixi run python examples/inference/3_predict.py
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)

# --- relational tables -> DuckDB --------------------------------------------
customers = pd.DataFrame(
    {
        "customer_id": range(20),
        "signup_time": pd.date_range("2020-01-01", periods=20, freq="7D"),
        "region": rng.choice(["us", "eu", "apac"], 20),
    }
)
orders = pd.DataFrame(
    {
        "order_id": range(100),
        "customer_id": rng.integers(0, 20, 100),
        "order_time": pd.date_range("2021-01-01", periods=100, freq="1D"),
        "amount": rng.gamma(2, 50, 100).round(2),
    }
)

db_path = HERE / "shop.duckdb"
db_path.unlink(missing_ok=True)
con = duckdb.connect(str(db_path))
con.execute("CREATE TABLE customers AS SELECT * FROM customers")
con.execute("CREATE TABLE orders AS SELECT * FROM orders")
con.close()

# --- labeled task split tables (binary churn) -------------------------------
for split, n in [("train", 60), ("val", 20), ("test", 20)]:
    pd.DataFrame(
        {
            "customer_id": rng.integers(0, 20, n),
            "timestamp": pd.date_range("2022-01-01", periods=n, freq="1D"),
            "churn": rng.integers(0, 2, n),
        }
    ).to_parquet(HERE / f"churn_{split}.parquet")

print(f"wrote {db_path} and churn_[train|val|test].parquet in {HERE}")
print("next: pixi run python examples/inference/1_data_prep.py")
