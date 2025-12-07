from huggingface_hub import HfApi, create_repo,login
import os

# Variables
repo_id = "Jagadesswar/tourism-prediction"
repo_type = "dataset"  # Can be "model" or "dataset" depending on your use case
# Retrieve Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("❌ Error: HF_TOKEN environment variable is not set.")
    exit(1)

# Login to Hugging Face Hub using token
try:
    login(token=hf_token)  # Login using the token
    print("✅ Logged into HuggingFace successfully.")
except Exception as e:
    print(f"❌ Error logging in to HuggingFace: {e}")
    exit(1)

# Initialize the API client
api = HfApi()

# Step 1: Check if the repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repository '{repo_id}' not found. Creating new repository...")
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repository '{repo_id}' created.")
    except Exception as e:
        print(f"❌ Error creating the repository: {e}")
        exit(1)

# Step 2: Upload the folder to Hugging Face Hub
folder_path = "tourism_project/data"  # Ensure this path is correct and relative to your script

# Check if the folder exists before uploading
if os.path.exists(folder_path):
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ Folder '{folder_path}' uploaded to Hugging Face successfully.")
    except Exception as e:
        print(f"❌ Error uploading folder to Hugging Face: {e}")
else:
    print(f"❌ Folder '{folder_path}' does not exist.")
