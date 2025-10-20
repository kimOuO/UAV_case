#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train YOLO on VisDrone (GPU if available), then auto-run validation and prediction,
and plot training/validation total loss curves.

Usage (Windows PowerShell, from the folder containing this file):
  python .\train_visdrone.py --data_yaml .\data.yaml --only_human --model yolov8n.pt --epochs 100 --imgsz 640 --name visdrone_humans

Notes
- VisDrone test-dev/test-challenge have NO labels, so test loss is not available.
  We therefore plot: training total loss vs. validation (testing proxy) total loss.
"""

import argparse
from pathlib import Path
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def make_human_only_yaml(src_yaml: Path, dst_yaml: Path):
    """Create a copy of data.yaml that keeps only 'pedestrian' and 'people' classes."""
    with open(src_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["nc"] = 2
    data["names"] = ["pedestrian", "people"]
    with open(dst_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def robust_sum(row, cols):
    s = 0.0
    for c in cols:
        if c in row and pd.notna(row[c]):
            s += float(row[c])
    return s

def auto_device(user_device: str) -> str:
    """Return a YOLO device string. If user passed 'cpu'/'0'/etc. respect it.
    If user passed 'auto', pick '0' if CUDA is available else 'cpu'."""
    if user_device.lower() != "auto":
        return user_device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def main():
    ap = argparse.ArgumentParser(description="Train YOLO on VisDrone and auto-run val/predict + plot losses.")
    ap.add_argument("--data_yaml", type=str, default="./data.yaml", help="Path to data.yaml")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics checkpoint, e.g., yolov8n.pt / yolov8m.pt")
    ap.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    ap.add_argument("--imgsz", type=int, default=640, help="Training image size")
    ap.add_argument("--device", type=str, default="auto", help="'auto' or explicit like '0'/'cpu'")
    ap.add_argument("--project", type=str, default="runs", help="Ultralytics project directory")
    ap.add_argument("--name", type=str, default="visdrone_yolov8n", help="Run name under project/detect/")
    ap.add_argument("--only_human", action="store_true", help="Keep only 'pedestrian' and 'people' classes")
    ap.add_argument("--predict_source", type=str, default="", help="Folder to run visual prediction. Default: val/images from YAML")
    args = ap.parse_args()

    data_yaml_path = Path(args.data_yaml).resolve()
    if not data_yaml_path.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml_path}")
        sys.exit(1)

    # Optionally build human-only YAML
    used_yaml = data_yaml_path
    tmp_human_yaml = None
    if args.only_human:
        tmp_human_yaml = data_yaml_path.parent / "data_human.yaml"
        make_human_only_yaml(data_yaml_path, tmp_human_yaml)
        used_yaml = tmp_human_yaml
        print(f"[INFO] Using human-only YAML: {used_yaml}")

    # Decide device
    device = auto_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Import YOLO
    try:
        from ultralytics import YOLO
    except Exception:
        print("[ERROR] Ultralytics not installed. Please run: pip install ultralytics")
        raise

    # Train
    model = YOLO(args.model)
    print(f"[INFO] Start training: model={args.model}, epochs={args.epochs}, imgsz={args.imgsz}, device={device}")
    _ = model.train(
        data=str(used_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.name,
        plots=True,
        verbose=True,
    )

    # Paths
    run_dir = Path(args.project) / "detect" / args.name
    results_csv = run_dir / "results.csv"

    # --- Auto validation (computes metrics, also writes val curves into results.csv)
    print("[INFO] Running validation (val)…")
    _ = model.val(data=str(used_yaml), imgsz=args.imgsz, device=device, project=args.project, name=args.name)

    # --- Auto prediction on val images (or user-provided source)
    try:
        with open(used_yaml, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        val_images_rel = y.get("val", "")
        # If YAML uses relative path to 'path', join them
        base = y.get("path", "")
        if base and isinstance(base, str) and not val_images_rel.startswith(("/", "\\")) and ":\\" not in val_images_rel:
            predict_source = str(Path(base) / val_images_rel)
        else:
            predict_source = val_images_rel
    except Exception:
        predict_source = ""

    if args.predict_source:
        predict_source = args.predict_source

    if predict_source:
        print(f"[INFO] Running visual prediction on: {predict_source}")
        _ = model.predict(
            source=predict_source,
            device=device,
            project=args.project,
            name=f"{args.name}_predict",
            save=True,
            imgsz=args.imgsz,
            stream=False
        )
    else:
        print("[WARN] Could not resolve a prediction source. Skipping prediction.")

    # --- Plot losses (training vs validation total loss)
    if not results_csv.exists():
        # Fallback to latest results.csv under project/detect
        cand = list((Path(args.project) / "detect").rglob("results.csv"))
        if cand:
            results_csv = max(cand, key=lambda p: p.stat().st_mtime)
            print(f"[WARN] Default results.csv not found, using latest: {results_csv}")
        else:
            print("[ERROR] results.csv not found. Cannot plot losses.")
            sys.exit(2)

    print(f"[INFO] Parsing: {results_csv}")
    df = pd.read_csv(results_csv)

    # Columns are version-dependent; discover loss columns robustly
    train_cols = [c for c in df.columns if c.startswith("train/") and c.endswith("_loss")]
    val_cols   = [c for c in df.columns if c.startswith("val/") and c.endswith("_loss")]
    if not train_cols:
        train_cols = [c for c in df.columns if c.startswith("train") and c.endswith("loss")]
    if not val_cols:
        val_cols   = [c for c in df.columns if c.startswith("val") and c.endswith("loss")]

    if not train_cols:
        print(f"[WARN] No train loss columns found. Columns: {list(df.columns)}")
    if not val_cols:
        print(f"[WARN] No val loss columns found. Columns: {list(df.columns)}")

    epochs = df.index.to_list()
    train_total = [robust_sum(r, train_cols) for _, r in df.iterrows()]
    val_total   = [robust_sum(r, val_cols)   for _, r in df.iterrows()]

    loss_df = pd.DataFrame({"epoch": epochs, "train_total_loss": train_total, "val_total_loss": val_total})
    loss_df.to_csv(run_dir / "loss_summary.csv", index=False)

    plt.figure()
    plt.plot(loss_df["epoch"], loss_df["train_total_loss"], label="training total loss")
    plt.plot(loss_df["epoch"], loss_df["val_total_loss"], label="validation total loss")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("Training vs Validation Total Loss (YOLO)")
    plt.legend(); plt.grid(True)
    plot_path = run_dir / "loss_curve.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"[INFO] Saved loss curve: {plot_path}")

    # Quick summary
    if len(loss_df):
        min_val_idx = loss_df["val_total_loss"].idxmin()
        print(f"[SUMMARY] Min val loss: {loss_df.loc[min_val_idx,'val_total_loss']:.4f} at epoch {loss_df.loc[min_val_idx,'epoch']}")
    print(f"[DONE] Run folder: {run_dir}")

if __name__ == "__main__":
    main()
