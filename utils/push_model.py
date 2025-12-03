from huggingface_hub import HfApi

api = HfApi(token="")
api.upload_folder(
    folder_path="",
    repo_id="",
    repo_type="model",
)
