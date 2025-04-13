import os
from pathlib import Path
import yaml
from collections import defaultdict
import shutil
import random

def limit_dataset_by_class(
    dataset_path: str,
    output_path: str,
    samples_per_class: int = 100
) -> dict:
    """
    Limit the number of images per class in YOLO dataset
    
    Args:
        dataset_path: Path to original YOLO dataset
        output_path: Path to save the limited dataset
        samples_per_class: Number of samples to keep per class
        
    Returns:
        dict: Statistics about the dataset reduction
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Read data.yaml
    with open(dataset_path / "data.yaml", "r") as f:
        data_config = yaml.safe_load(f)
        
    # Create output directories
    for split in ["train", "valid"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Copy data.yaml
    shutil.copy2(dataset_path / "data.yaml", output_path / "data.yaml")
    
    stats = {
        "original_counts": defaultdict(int),
        "limited_counts": defaultdict(int),
        "removed_images": 0
    }
    
    # Track images per class
    class_images = defaultdict(list)
    
    # Scan dataset and group images by class
    for split in ["train", "valid"]:
        label_dir = dataset_path / split / "labels"
        image_dir = dataset_path / split / "images"
        
        if not label_dir.exists() or not image_dir.exists():
            continue
            
        for label_file in label_dir.glob("*.txt"):
            image_stem = label_file.stem
            
            # Find corresponding image
            image_file = None
            for ext in [".jpg", ".jpeg", ".png"]:
                if (image_dir / f"{image_stem}{ext}").exists():
                    image_file = image_dir / f"{image_stem}{ext}"
                    break
            
            if not image_file:
                continue
                
            # Read label file to determine classes
            with open(label_file, "r") as f:
                classes_in_image = set()
                for line in f:
                    class_id = int(line.split()[0])
                    classes_in_image.add(class_id)
                    stats["original_counts"][class_id] += 1
            
            # Store image information for each class it contains
            for class_id in classes_in_image:
                class_images[class_id].append({
                    "image": image_file,
                    "label": label_file,
                    "split": split
                })

    # Select limited samples for each class
    selected_images = set()
    for class_id, images in class_images.items():
        # Randomly select samples_per_class images
        selected = random.sample(images, min(samples_per_class, len(images)))
        for item in selected:
            selected_images.add(item["image"])
            stats["limited_counts"][class_id] += 1

    # Copy selected images and their labels
    for split in ["train", "valid"]:
        for image_path in (dataset_path / split / "images").glob("*.*"):
            if image_path in selected_images:
                # Copy image
                shutil.copy2(
                    image_path,
                    output_path / split / "images" / image_path.name
                )
                # Copy corresponding label
                label_path = dataset_path / split / "labels" / f"{image_path.stem}.txt"
                if label_path.exists():
                    shutil.copy2(
                        label_path,
                        output_path / split / "labels" / f"{image_path.stem}.txt"
                    )
            else:
                stats["removed_images"] += 1

    return stats

def main():
    dataset_path = "../data/raw"  # 元のデータセットパス
    output_path = "../setup_data/source_yolo_dataset"  # 制限されたデータセットの出力パス
    samples_per_class = 100
    
    stats = limit_dataset_by_class(
        dataset_path=dataset_path,
        output_path=output_