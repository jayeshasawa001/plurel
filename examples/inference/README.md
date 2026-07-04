# Run PluRel pretrained RT on your own database

Run a pretrained checkpoint on your own database (DuckDB, Postgres, or MySQL) in
three steps.

You edit exactly one file, [`config.py`](config.py), then run three scripts in order:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `1_data_prep.py` | Convert your database into RelBench format (declare primary keys, foreign keys, time columns). |
| 2 | `2_task_prep.py` | Define the prediction task (entity, timestamp, target) and its labeled train/val/test rows. |
| 3 | `3_predict.py` | Download the pretrained model, preprocess, run inference, and report the metric. |

## Try it first on a demo database

`config.py` ships pre-filled with a tiny two-table shop database, so you can run the
whole flow before touching your own data. From the repo root (with the [Setup](../../README.md#setup)
environment and the compiled Rust sampler):

```bash
pixi run python examples/inference/make_demo_duckdb.py   # build the sample DuckDB
pixi run python examples/inference/1_data_prep.py
pixi run python examples/inference/2_task_prep.py
pixi run python examples/inference/3_predict.py          # add --device cpu if no GPU
```

### What you should see

Step 1 converts the SQL tables to RelBench format:

```text
[step 1] reading tables from .../examples/inference/shop.duckdb
[step 1] wrote RelBench database 'shop-demo' -> ~/scratch/relbench/shop-demo/db
   - customers: 20 rows | pkey=customer_id | fkeys={} | time=signup_time
   - orders: 100 rows | pkey=order_id | fkeys={'customer_id': 'customers'} | time=order_time
```

Step 2 writes the labeled task tables:

```text
[step 2] task 'customer-churn' (binary_classification)
   predict 'churn' for 'customers' at 'timestamp'
   - train: 60 labeled rows
   - val: 20 labeled rows
   - test: 20 labeled rows
[step 2] wrote task tables -> ~/scratch/relbench/shop-demo/tasks/customer-churn
```

Step 3 downloads the pretrained Relational Transformer checkpoint, preprocesses the
database into the model's tensor format (one-time; cached afterwards), runs zero-shot
inference over the test split, and reports the metric:

```text
[step 3] resolving checkpoint
   downloading checkpoint stanford-star/rt-plurel/synthetic-pretrain_rdb_1024_size_4b.pt from the HF Hub...
[step 3] preprocessing
   preprocessing 'shop-demo' with the Rust sampler...
   embedding text columns for 'shop-demo'...
[step 3] running inference on shop-demo/customer-churn (device=cpu)
   batch 1/1 (112s)

[result] shop-demo/customer-churn   AUROC = 0.4115   (n=20)
```

> The demo's labels are random noise, so chance-level AUROC (~0.5) is expected —
> the demo exercises the pipeline, not the model.

## Point it at your own database

Open [`config.py`](config.py) and set:

1. **`SQL_URI`** — how to reach your database:
   - DuckDB: a `.duckdb` file path
   - Postgres: `postgresql+psycopg2://user:pw@host:5432/dbname` (install `psycopg2-binary`)
   - MySQL: `mysql+pymysql://user:pw@host:3306/dbname` (install `pymysql`)

2. **`TABLES`** — your schema: for each table, its primary key, time column, and
   foreign keys (`{column: referenced_table}`).

3. **`TASK`** — what to predict: the entity table/column, the timestamp column, the
   target column, the task type (`binary_classification` or `regression`), and the
   paths to your labeled `train` / `val` / `test` rows (parquet or csv, each with the
   entity column, time column, and target column). Only `test` is required.

Then rerun the three steps. That's it.
