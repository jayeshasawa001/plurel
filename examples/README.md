# Examples

- [`generation/`](generation/) — synthesize a synthetic relational database.
  - `synthesize_from_sql.ipynb`: generate a database from a SQL schema.

- [`inference/`](inference/) — run a pretrained checkpoint on your own database.
  A guided three-step walkthrough (`1_data_prep` → `2_task_prep` → `3_predict`) for
  using DuckDB / Postgres / MySQL data with a PluRel model. See
  [`inference/README.md`](inference/README.md).

  ```bash
  pixi run python examples/inference/make_demo_duckdb.py
  pixi run python examples/inference/1_data_prep.py
  pixi run python examples/inference/2_task_prep.py
  pixi run python examples/inference/3_predict.py
  ```
