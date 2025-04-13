import kagglehub
import os
import shutil
from pathlib import Path

def download_and_prepare_dataset(
    dataset_name: str = "rezeliet/yolo-vegetable-dataset",
    output_dir: str = "../data/raw"
) -> dict:
    """
    Function to download a Kaggle dataset and prepare the directory structure

    Args:
        dataset_name (str): Name of the Kaggle dataset to download
        output_dir (str): Dir pass for output directory

    Returns:
        dict: dictionary containing paths to the dataset directories
    """
    try:
        # dataset_name = dataset_name
        download_path = kagglehub.dataset_download(dataset_name)
        
        # 出力ディレクトリの作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # ダウンロードしたファイルを出力ディレクトリにコピー
        for file_path in Path(download_path).glob('**/*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(download_path)
                destination = output_path / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, destination)
        
        # データセットの構造を確認
        image_dir = output_path / "images"
        label_dir = output_path / "labels"
        
        dataset_info = {
            "base_path": str(output_path),
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "download_path": download_path
        }
        
        print(f"データセットを {output_path} に準備しました")
        return dataset_info
    
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    # 関数のテスト実行
    dataset_info = download_and_prepare_dataset()
    print("\nデータセット情報:")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")