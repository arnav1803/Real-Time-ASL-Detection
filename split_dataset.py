import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, target_dir, split_ratio=(0.7, 0.15, 0.15)):
    train_ratio, test_ratio, val_ratio = split_ratio
    assert train_ratio + test_ratio + val_ratio == 1.0, "Split ratios must sum to 1"

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directories
    for split in ['train', 'test', 'val']:
        (target_path / split).mkdir(parents=True, exist_ok=True)

    # Iterate over classes (A-Z)
    classes = [d.name for d in source_path.iterdir() if d.is_dir()]

    print(f"Found classes: {classes}")

    for class_name in classes:
        class_source_dir = source_path / class_name
        files = [f for f in class_source_dir.iterdir() if f.is_file()]
        random.shuffle(files)

        total_files = len(files)
        train_count = int(total_files * train_ratio)
        test_count = int(total_files * test_ratio)
        # val_count takes the remainder to ensure no files are left behind due to rounding

        train_files = files[:train_count]
        test_files = files[train_count:train_count + test_count]
        val_files = files[train_count + test_count:]

        print(
            f"Class {class_name}: Total {total_files} -> Train {len(train_files)}, Test {len(test_files)}, Val {len(val_files)}")

        # Copy files
        splits = {
            'train': train_files,
            'test': test_files,
            'val': val_files
        }

        for split_name, split_files in splits.items():
            split_dir = target_path / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for file in split_files:
                shutil.copy2(file, split_dir / file.name)


if __name__ == "__main__":
    # Adjust paths as needed.
    # Assuming script is run from C:\Users\arnav\Coding\ASL
    source = "signalphaset"
    target = "data"

    if not os.path.exists(source):
        print(f"Error: Source directory '{source}' not found.")
    else:
        split_dataset(source, target)
        print("Dataset split completed successfully.")
