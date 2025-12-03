from huggingface_hub import HfApi

api = HfApi(token="")
api.upload_folder(
    folder_path="data/",
    repo_id="",
    repo_type="dataset",
)

