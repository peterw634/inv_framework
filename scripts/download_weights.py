"""Download the pretrained weights used by the learned algorithms into weights/.

    python scripts/download_weights.py                 # everything (~1.9 GB)
    python scripts/download_weights.py pmf             # NaSn pMF only (502 MiB)
    python scripts/download_weights.py lodoind-pixel lodoind-latent
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_weights.py lodoind-pixel

Downloads honour the usual HTTP(S)_PROXY variables. From mainland China a
direct connection to the mirror is usually much faster than a proxy (measured
here: ~78 KB/s through the proxy, >1.5 MB/s direct):

    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
        HF_ENDPOINT=https://hf-mirror.com python scripts/download_weights.py

The pMF weights come from GitHub releases (github.com); keep the proxy for those
if GitHub is not reachable directly.

Revisions are pinned; the pMF files are checked against the SHA-256 values of
the model manifest. Files that are already present and verified are skipped.
"""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "inv_framework/applications/nasn_ct/manifests/model-v0.1.0.json"

HF_MODELS = {
    # DM4CT priors trained on LoDoInd (industrial CT), MIT licence.
    "lodoind-pixel": ("jiayangshi/lodoind_pixel_diffusion", "0f4a0dcf5768abec78fdbc01b95a83194175c0f1"),
    "lodoind-latent": ("jiayangshi/lodoind_latent_diffusion", "6f35a9b0198a3a7e6eec64b69112745b3a24a295"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pmf(target):
    manifest = json.loads(MANIFEST.read_text())
    target.mkdir(parents=True, exist_ok=True)
    for asset in manifest["assets"].values():
        path = target / asset["filename"]
        if path.exists() and sha256(path) == asset["sha256"]:
            print(f"ok       {path}")
            continue
        print(f"download {asset['url']}")
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(asset["url"]) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out, length=8 << 20)
        actual = sha256(partial)
        if actual != asset["sha256"]:
            partial.unlink()
            raise SystemExit(f"SHA-256 mismatch for {asset['filename']}: {actual}")
        partial.replace(path)
        print(f"ok       {path}")


def fetch_hf(name, target):
    from huggingface_hub import snapshot_download
    repo, revision = HF_MODELS[name]
    path = snapshot_download(repo, revision=revision, local_dir=target / repo.split("/")[1])
    print(f"ok       {path} ({repo}@{revision[:7]})")


def main():
    choices = ["pmf", *HF_MODELS]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # No argparse choices: Python 3.10 rejects an empty nargs="*" list against them.
    parser.add_argument("models", nargs="*", metavar="MODEL",
                        help=f"models to fetch, any of {choices} (default: all)")
    parser.add_argument("--output", type=Path, default=ROOT / "weights")
    args = parser.parse_args()
    unknown = sorted(set(args.models) - set(choices))
    if unknown:
        parser.error(f"unknown model(s) {unknown}; choose from {choices}")
    for name in args.models or choices:
        if name == "pmf":
            fetch_pmf(args.output)
        else:
            fetch_hf(name, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
