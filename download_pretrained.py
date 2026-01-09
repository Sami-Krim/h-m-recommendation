import gdown
import os

# File mappings: filename -> (file_id, output_directory)
FILES = {
    "hm_pretokenized_data.pt": ("1z7rUsZSN51lyaILyA3-2qz-v6ts4YpMV", "recommender_lora/pretrained"),
    "refined_article_embeddings.npy": ("1b1OY31z40OtkGwqnLmBcV3a79JAu5yft", "recommender_lora/pretrained"),
    "best_hybrid_vae.pt": ("1EIViBwAtuq6JWStj1J_rBZ83-9MoX_v8", "recommender_vae/pretrained"),
    "qwen_embeddings_full.pt": ("1D5Gm3XJBPNAyZ0e403okkrBZV2J7Tvvw", "recommender_vae/pretrained")
}

# Folder mappings: folder_name -> (folder_id, output_directory)
FOLDERS = {
    "lora_epoch_4": ("19ZFALBMZgcq5ySIu3yAT0IEsR2xud3Wk", "recommender_lora/pretrained"),
}

def download_file(file_id, output_path):
    """
    Download a single file from Google Drive.
    """

    # Clean up any incomplete downloads (.part files)
    output_dir = os.path.dirname(output_path)
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith(os.path.basename(output_path)) and file.endswith('.part'):
                part_file = os.path.join(output_dir, file)
                print(f"Cleaning up incomplete download: {file}")
                try:
                    os.remove(part_file)
                except Exception as e:
                    print(f"   Warning: Could not remove {file}: {e}")
    
    if os.path.exists(output_path):
        # Verify file is not empty
        if os.path.getsize(output_path) > 0:
            print(f"{os.path.basename(output_path)} already exists.")
            return True
        else:
            print(f"{os.path.basename(output_path)} exists but is empty, re-downloading...")
            os.remove(output_path)
    
    print(f"\nDownloading {os.path.basename(output_path)}...")
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        # Use fuzzy=True to handle large files better
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        
        # Verify download completed successfully
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Successfully downloaded {os.path.basename(output_path)}")
            return True
        else:
            print(f"Download appears incomplete (file size: {os.path.getsize(output_path) if os.path.exists(output_path) else 0})")
            return False
    except Exception as e:
        print(f"Error downloading {os.path.basename(output_path)}: {e}")
        # Clean up any partial files
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return False

def download_folder(folder_id, output_dir):
    """
    Download a folder from Google Drive.
    """

    # Check if folder exists and has expected files
    if os.path.exists(output_dir):
        expected_files = ['adapter_config.json', 'adapter_model.safetensors']
        existing_files = os.listdir(output_dir)
        if all(f in existing_files for f in expected_files):
            print(f"{os.path.basename(output_dir)} folder already exists with required files.")
            return True
    
    print(f"\nDownloading folder {os.path.basename(output_dir)}...")
    url = f'https://drive.google.com/drive/folders/{folder_id}'
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
        print(f"Successfully downloaded {os.path.basename(output_dir)} folder")
        return True
    except Exception as e:
        print(f"Error downloading folder {os.path.basename(output_dir)}: {e}")
        print(f"   You may need to download it manually from: {url}")
        return False

def download_data(project_root="."):
    """
    Download all required files and folders from Google Drive.
    
    Args:
        project_root: Root directory of the project (default: current directory)
    """

    print("Starting data download ...")
    
    # Download individual files
    for filename, (file_id, subdir) in FILES.items():
        output_dir = os.path.join(project_root, subdir)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        download_file(file_id, output_path)
    
    # Download folders
    for folder_name, (folder_id, subdir) in FOLDERS.items():
        output_dir = os.path.join(project_root, subdir, folder_name)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        download_folder(folder_id, output_dir)
    
    print("\nDownload complete!")

if __name__ == "__main__":
    download_data()