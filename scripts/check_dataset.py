import yaml
from pathlib import Path
cfg_path = Path("configs/dataset.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
root = (cfg_path.parent / cfg["path"]).resolve()
errors = []
for split in ("train", "val"):
    for kind in ("images", "labels"):
        folder = root / kind / split
        if not folder.exists(): errors.append(f"Missing {folder}")
if errors:
    raise SystemExit("\n".join(errors))
print("Dataset directories found. Confirm labels, class IDs and image-label pairing.")
