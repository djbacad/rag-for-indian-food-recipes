import kaggle

# Set your Kaggle API credentials
# Ensure you have the json file containing the credentials.
kaggle.api.authenticate()

# Dataset spec
dataset_name = 'kanishk307/6000-indian-food-recipes-dataset' 
dataset_version = 'latest'  # 'latest' to get the most recent version
download_path = '../datasets'

# Download the dataset
kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
print(f"Dataset '{dataset_name}' downloaded successfully to '{download_path}'")

