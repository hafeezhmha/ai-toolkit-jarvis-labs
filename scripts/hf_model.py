from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/home/ai-toolkit/output/model",
    repo_id="username/model_name",
    repo_type="model",
)
