from huggingface_hub import hf_hub_download


def download_from_hf(
    filename: str,
) -> str:
    """
    Download a file from DFoT Hugging Face model hub.
    https://huggingface.co/kiwhansong/DFoT
    """
    return hf_hub_download(
        repo_id="kiwhansong/DFoT",
        cache_dir="./huggingface",
        filename=filename,
    )
