from typing import Optional
import os
import sys
import warnings
from urllib.parse import urlparse  # noqa: F401
from torch.hub import get_dir, HASH_REGEX, download_url_to_file


def download_model_from_url(
    url: str,
    model_dir: Optional[str] = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
) -> str:
    r"""Download a file containing a pre-trained model from a URL. Supports automatic caching at torch.hub's cache directory.
    Adapted from `torch.hub.load_state_dict_from_url`.

    Args:
        url (str): URL of the object to download
        model_dir (str, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.
            Recommended for untrusted sources. See :func:`~torch.load` for more details.
    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead"
        )

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file
