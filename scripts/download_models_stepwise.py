import os
import sys
import argparse
import subprocess

MEDIUM_MODELS = [
    "Helsinki-NLP/opus-mt-zh-en",
    "openai/clip-vit-base-patch32",
    "Salesforce/blip-image-captioning-large",
]

LARGE_MODELS = [
    "sd2-community/stable-diffusion-2-1-base",
    "runwayml/stable-diffusion-v1-5",
    "lllyasviel/sd-controlnet-canny",
]


def run_one(py_exe: str, model_id: str, timeout_sec: int, token: str | None) -> bool:
    code = (
        "from huggingface_hub import snapshot_download\n"
        f"p = snapshot_download(repo_id={model_id!r}, token={token!r})\n"
        "print('SNAPSHOT_PATH=' + p)\n"
    )

    cmd = [py_exe, "-u", "-c", code]
    print(f"\n=== Download: {model_id} | timeout={timeout_sec}s ===", flush=True)
    print("CMD:", " ".join(cmd[:3]) + " ...", flush=True)

    # Inherit stdout/stderr (do not PIPE). huggingface_hub uses tqdm with \\r updates;
    # with PIPE, the parent only reads on newlines so the console looks "stuck".
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {model_id} 超过 {timeout_sec}s，已中断", flush=True)
        return False

    if result.returncode == 0:
        print(f"OK: {model_id}", flush=True)
        return True
    print(f"FAIL: {model_id} | returncode={result.returncode}", flush=True)
    return False


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Step-by-step HF model downloader with timeout")
    parser.add_argument("--python", required=True, help="Python executable path, e.g. E:/Course/Statistic_L/.venv-gpu/Scripts/python.exe")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per model in seconds")
    parser.add_argument("--group", choices=["medium", "large", "all"], default="medium")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or None
    print("HF_TOKEN exists:", bool(token), flush=True)

    models = []
    if args.group in ("medium", "all"):
        models.extend(MEDIUM_MODELS)
    if args.group in ("large", "all"):
        models.extend(LARGE_MODELS)

    ok = 0
    for m in models:
        if run_one(args.python, m, args.timeout, token):
            ok += 1

    print(f"\nSummary: {ok}/{len(models)} success", flush=True)
    return 0 if ok == len(models) else 2


if __name__ == "__main__":
    sys.exit(main())
