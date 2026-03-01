
"""
Train ONLY the UNSTAINED dataset using your existing train.py.

This wrapper filters BOTH:
- data/labels_points
- data/augmented

It temporarily keeps only folders containing "unstained",
runs training, then restores everything.

Project structure expected:

viscell/
    src/ai/train.py
    data/
        labels_points/
            batch_A/stained
            batch_A/unstained
            ...
        augmented/
            batch_A/stained
            batch_A/unstained
            ...
"""

from pathlib import Path
import shutil
import subprocess
import sys


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def filter_unstained(src: Path, dst: Path):
    for p in src.rglob("*"):
        if "unstained" not in [part.lower() for part in p.parts]:
            continue
        rel = p.relative_to(src)
        out = dst / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def main():
    root = project_root()
    data_dir = root / "data"

    labels = data_dir / "labels_points"
    augmented = data_dir / "augmented"

    if not labels.exists() or not augmented.exists():
        print("[ERROR] data/labels_points or data/augmented not found")
        return 1

    backup_labels = data_dir / "_backup_labels_points"
    backup_aug = data_dir / "_backup_augmented"

    temp_labels = data_dir / "_temp_labels_points"
    temp_aug = data_dir / "_temp_augmented"

    if backup_labels.exists() or backup_aug.exists():
        print("[ERROR] Backup folders already exist. Delete manually if safe.")
        return 2

    try:
        print("[INFO] Backing up original folders")
        labels.rename(backup_labels)
        augmented.rename(backup_aug)

        print("[INFO] Creating unstained-only datasets")
        temp_labels.mkdir()
        temp_aug.mkdir()

        filter_unstained(backup_labels, temp_labels)
        filter_unstained(backup_aug, temp_aug)

        temp_labels.rename(labels)
        temp_aug.rename(augmented)

        train_py = root / "src" / "ai" / "train.py"

        cmd = [
            sys.executable,
            str(train_py),
            "--data",
            "data",
            "--run_name",
            "viscell_unet_512_cpu_unstained"
        ]

        print("[INFO] Starting training:")
        print("       " + " ".join(cmd))

        subprocess.call(cmd, cwd=str(root))

    finally:
        print("[INFO] Restoring original folders")

        if labels.exists():
            shutil.rmtree(labels)
        if augmented.exists():
            shutil.rmtree(augmented)

        if backup_labels.exists():
            backup_labels.rename(labels)
        if backup_aug.exists():
            backup_aug.rename(augmented)

        if temp_labels.exists():
            shutil.rmtree(temp_labels)
        if temp_aug.exists():
            shutil.rmtree(temp_aug)

        print("[INFO] Restore complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
