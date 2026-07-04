"""Describe YOUR database and prediction task here — this is the only file you edit.

It ships pre-filled with the demo database built by ``make_demo_duckdb.py`` so the
walkthrough runs out of the box. Point it at your own database and change the schema
+ task to run on your data. Then run, in order:

    1_data_prep.py   # convert your database to RelBench format
    2_task_prep.py   # define the prediction task
    3_predict.py     # download the model + predict + score
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent

# A short name for this dataset (becomes a directory under the relbench cache).
DB_NAME = "shop-demo"

# --- how to reach your database ---------------------------------------------
# DuckDB   : a .duckdb file path (the demo default)
# Postgres : "postgresql+psycopg2://user:password@host:5432/dbname"
# MySQL    : "mysql+pymysql://user:password@host:3306/dbname"
# (install psycopg2-binary / pymysql for Postgres / MySQL.)
SQL_URI = str(_HERE / "shop.duckdb")

# --- your relational schema --------------------------------------------------
# One entry per table you want to include. For each:
#   pkey     : the primary-key column
#   time_col : the row-timestamp column (or None if the table has no time)
#   fkeys    : {foreign_key_column: table_it_points_to}
TABLES = {
    "customers": {
        "pkey": "customer_id",
        "time_col": "signup_time",
        "fkeys": {},
    },
    "orders": {
        "pkey": "order_id",
        "time_col": "order_time",
        "fkeys": {"customer_id": "customers"},  # orders.customer_id -> customers
    },
}

# --- your prediction task ----------------------------------------------------
# You predict `target_col` for an `entity_table` entity as of a given time.
# The split tables are the LABELED rows: each has the entity column, the time
# column, and the target column (parquet or csv). `test` is required; `train`
# is used for regression target normalization when present.
TASK = {
    "name": "customer-churn",
    "entity_table": "customers",
    "entity_col": "customer_id",
    "time_col": "timestamp",
    "target_col": "churn",
    "task_type": "binary_classification",  # or "regression"
    "splits": {
        "train": str(_HERE / "churn_train.parquet"),
        "val": str(_HERE / "churn_val.parquet"),
        "test": str(_HERE / "churn_test.parquet"),
    },
}

# --- pretrained checkpoint ---------------------------------------------------
# Downloaded from the Hugging Face Hub. Set CHECKPOINT to a local .pt path to
# skip the download.
HF_REPO = "stanford-star/rt-plurel"
HF_FILENAME = "synthetic-pretrain_rdb_1024_size_4b.pt"
CHECKPOINT = None  # e.g. "~/scratch/rt_hf_ckpts/synthetic-pretrain_rdb_1024_size_4b.pt"
