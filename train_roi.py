# train_roi.py
import argparse, os
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser(description="Train ROI detector for RU passport series/number")
    ap.add_argument("--data", default="datasets/passport_roi/data.yaml",
                    help="Path to data.yaml")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Base model: yolov8n.pt | yolov8s.pt | path/to/.pt")
    ap.add_argument("--imgsz", type=int, default=896,
                    help="Image size for training/val (e.g., 896 or 1024)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Number of epochs")
    ap.add_argument("--batch", default=-1, type=int,
                    help="Batch size (-1 to auto-fit)")
    ap.add_argument("--device", default="0",
                    help='CUDA device id, e.g. "0"; use "cpu" to force CPU')
    ap.add_argument("--workers", type=int, default=2,
                    help="Dataloader workers")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # quick dataset sanity checks
    data_yaml = Path(args.data)
    root = data_yaml.parent if data_yaml.exists() else None
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        p = root / sub
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset folder: {p}")
        if "images" in sub and not any(p.iterdir()):
            raise RuntimeError(f"No images in: {p}")
        if "labels" in sub and not any(p.iterdir()):
            raise RuntimeError(f"No labels in: {p}")

    print("✅ Dataset OK:", root.resolve())
    print("▶️  Training:", args.model, "epochs:", args.epochs, "imgsz:", args.imgsz, "device:", args.device)

    model = YOLO(args.model)

    train_res = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        lr0=0.01,
        warmup_epochs=3,
        patience=50,       # early stopping
        degrees=2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        mosaic=0.5,
        hsv_h=0.0,
        hsv_s=0.2,
        hsv_v=0.2,
    )

    # pick best.pt and run validation
    run_dir = Path(train_res.save_dir)  # runs/detect/trainX
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError("best.pt not found in run dir: " + str(run_dir))

    print("\n✅ Training finished. best.pt:", best.resolve())
    print("▶️  Validating best.pt ...")

    model = YOLO(str(best))
    val_res = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        device=args.device,
        split="val",
    )

    # Pretty print a few key metrics
    try:
        mAP50 = float(val_res.results_dict.get("metrics/mAP50(B)", 0.0))
        recall = float(val_res.results_dict.get("metrics/recall(B)", 0.0))
        precision = float(val_res.results_dict.get("metrics/precision(B)", 0.0))
        print(f"\n📊 Val metrics  mAP50: {mAP50:.3f}  |  Recall: {recall:.3f}  |  Precision: {precision:.3f}")
    except Exception:
        pass

    print("\n➡️  Use this model for inference:\n", best.resolve())

if __name__ == "__main__":
    main()
