import kagglehub

# Download latest version
path = kagglehub.dataset_download("rezeliet/yolo-vegetable-dataset")

print("Path to dataset files:", path)