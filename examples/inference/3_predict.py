"""Step 3 — Predict and score.

Downloads the pretrained checkpoint from the Hugging Face Hub, preprocesses the
database prepared in steps 1-2 into the model's tensor format (Rust sampler + text
embeddings), runs zero-shot inference on the test split, joins the predictions back
to your labeled rows, and reports the metric (AUROC for classification, MAE for
regression).

    pixi run python examples/inference/3_predict.py            # GPU (default)
    pixi run python examples/inference/3_predict.py --device cpu
"""

import argparse
import os

import config
import pipeline
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=32)
    # >0 needs fork-able dataloader workers (Linux); the Rust sampler can't be
    # pickled for spawn-based workers (macOS), so load in-process by default.
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    # flex_attention's torch.compile hangs on CPU; run it eager there.
    if args.device == "cpu":
        os.environ["RT_EAGER"] = "1"

    if not (pipeline.dataset_dir(config.DB_NAME) / "tasks" / config.TASK["name"]).exists():
        raise SystemExit("Run 1_data_prep.py and 2_task_prep.py first.")

    print("[step 3] resolving checkpoint")
    ckpt = pipeline.resolve_checkpoint(config.CHECKPOINT, config.HF_REPO, config.HF_FILENAME)

    print("[step 3] preprocessing")
    pipeline.ensure_preprocessed(config.DB_NAME, pipeline.MODEL_DEFAULTS["embedding_model"])

    print(
        f"[step 3] running inference on {config.DB_NAME}/{config.TASK['name']} (device={args.device})"
    )
    ent, ts, pred = pipeline.run_inference(
        config, ckpt, args.device, args.batch_size, args.num_workers
    )
    pipeline.evaluate(config, ent, ts, pred)


if __name__ == "__main__":
    main()
