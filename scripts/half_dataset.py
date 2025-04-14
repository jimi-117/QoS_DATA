import yaml
import os
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm
import sys
import logging
import random
from typing import Dict, List, Set

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Path Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "from_api"  # 元のデータセットの場所
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "half_dataset"  # 新しいデータセットの出力先

# --- Constants ---
RANDOM_SEED = 42  # 再現性のための固定シード
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

def setup_directories() -> bool:
    """出力ディレクトリの作成"""
    try:
        for split in ['train', 'val', 'test']:
            (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False

def read_yaml_config(yaml_path: Path) -> dict:
    """YAMLファイルの読み込み"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to read YAML file: {e}")
        return {}

def group_files_by_class(label_dir: Path) -> Dict[int, List[Path]]:
    """クラスごとにファイルをグループ化"""
    class_files: Dict[int, List[Path]] = {}
    
    for label_file in label_dir.glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if class_id not in class_files:
                        class_files[class_id] = []
                    if label_file not in class_files[class_id]:
                        class_files[class_id].append(label_file)
                    break  # 最初の行のクラスIDだけを使用
        except Exception as e:
            logger.warning(f"Error reading {label_file}: {e}")
            continue
    
    return class_files

def select_half_files(class_files: Dict[int, List[Path]]) -> Set[Path]:
    """各クラスから半分のファイルを選択"""
    random.seed(RANDOM_SEED)
    selected_files = set()
    
    for class_id, files in class_files.items():
        half_count = max(1, len(files) // 2)
        selected = random.sample(files, half_count)
        selected_files.update(selected)
        logger.info(f"Class {class_id}: Selected {half_count}/{len(files)} files")
    
    return selected_files

def process_dataset() -> bool:
    """メインの処理関数"""
    # ディレクトリの設定
    if not setup_directories():
        return False

    # 元のYAML設定の読み込み
    source_yaml = SOURCE_DIR / "data.yaml"
    if not source_yaml.exists():
        logger.error(f"Source data.yaml not found at {source_yaml}")
        return False

    yaml_config = read_yaml_config(source_yaml)
    if not yaml_config:
        return False

    # ファイルのグループ化と選択
    label_dir = SOURCE_DIR / "labels" / "train"
    class_files = group_files_by_class(label_dir)
    selected_files = select_half_files(class_files)

    # 選択されたファイルの処理
    processed_count = 0
    for label_file in tqdm(selected_files, desc="Processing files"):
        image_stem = label_file.stem
        
        # 対応する画像ファイルを検索
        image_file = None
        for ext in IMAGE_EXTENSIONS:
            potential_image = SOURCE_DIR / "images" / "train" / f"{image_stem}{ext}"
            if potential_image.exists():
                image_file = potential_image
                break

        if not image_file:
            logger.warning(f"No image found for {label_file}")
            continue

        try:
            # ラベルとイメージをコピー
            copyfile(label_file, OUTPUT_DIR / "labels" / "train" / label_file.name)
            copyfile(image_file, OUTPUT_DIR / "images" / "train" / image_file.name)
            processed_count += 1
        except Exception as e:
            logger.error(f"Error copying files for {label_file.stem}: {e}")
            continue

    # 新しいYAML設定の作成
    new_yaml_config = {
        'path': '..',
        'train': 'images/train',
        'val': 'images/train',
        'test': '',
        'nc': yaml_config.get('nc', 0),
        'names': yaml_config.get('names', [])
    }

    try:
        with open(OUTPUT_DIR / "data.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(new_yaml_config, f, sort_keys=False, default_flow_style=None, allow_unicode=True)
    except Exception as e:
        logger.error(f"Failed to write new data.yaml: {e}")
        return False

    logger.info(f"Successfully processed {processed_count} files")
    logger.info(f"Output dataset location: {OUTPUT_DIR}")
    return True

if __name__ == "__main__":
    if not process_dataset():
        sys.exit(1)