from huggingface_hub import HfApi,login
import os

hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)
api = HfApi()
print("✅ Logged into HuggingFace successfully.\n")
api.upload_folder(
    folder_path="deployment",     # the local folder containing your files
    repo_id="Jagadesswar/Tourism_Product_Purchase_Prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional:subfolder path inside the repo
)
