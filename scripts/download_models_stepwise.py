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

# SD1.5 + 参考模式：按需任选子集；Annotators 供 Depth/OpenPose/HED/Lineart 预处理
REFERENCE_SNAPSHOTS = [
    "Lykon/dreamshaper-8",
    "lllyasviel/sd-controlnet-depth",
    "lllyasviel/sd-controlnet-openpose",
    "lllyasviel/control_v11p_sd15_lineart",
    "lllyasviel/sd-controlnet-hed",
    "lllyasviel/sd-controlnet-canny",
    "lllyasviel/Annotators",
]

REFERENCE_SINGLE_FILES = [
    ("h94/IP-Adapter", "models/ip-adapter_sd15_light_v11.bin"),
]

# SD1.5 风格：DreamShaper + 油画 LoRA 仓库（snapshot 整库）
SD15_STYLE_MODELS = [
    "Lykon/dreamshaper-8",
    "jqlive/sd15-digital-oil-arcane",
]

# 仅下载单文件：避免 snapshot 拉取整库 / 水彩仓库内超长文件名预览图（Windows 下易 symlink 失败）
SD15_SINGLE_FILES = [
    (
        "WarriorMama777/OrangeMixs",
        "Models/AbyssOrangeMix3/AOM3A3_orangemixs.safetensors",
    ),
    ("fladdict/watercolor", "fladdict-watercolor-sd-1-5.safetensors"),
    ("jordanhilado/sd-1-5-sketch-lora", "pytorch_lora_weights.safetensors"),
]


def _extend_unique(dest: list, items: list) -> None:
    seen = set(dest)
    for x in items:
        if x not in seen:
            seen.add(x)
            dest.append(x)


def webapp_model_ids() -> list[str]:
    """文生图 + SD1.5 参考模式（含 controlnet-aux 用的 Annotators），不含可选二次元/DreamShaper 风格包。"""
    out: list[str] = []
    _extend_unique(out, MEDIUM_MODELS)
    _extend_unique(out, LARGE_MODELS)
    _extend_unique(out, REFERENCE_SNAPSHOTS)
    return out


def hf_download_env() -> dict[str, str]:
    """Merge env for child Python: longer Hub timeouts (default 10s often fails on slow links)."""
    env = os.environ.copy()
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    return env


def run_one_file(py_exe: str, repo_id: str, filename: str, timeout_sec: int, token: str | None) -> bool:
    code = (
        "from huggingface_hub import hf_hub_download\n"
        f"p = hf_hub_download(repo_id={repo_id!r}, filename={filename!r}, token={token!r})\n"
        "print('FILE_PATH=' + p)\n"
    )
    cmd = [py_exe, "-u", "-c", code]
    print(f"\n=== Download file: {repo_id}/{filename} | timeout={timeout_sec}s ===", flush=True)
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout_sec,
            check=False,
            env=hf_download_env(),
        )
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {repo_id}/{filename}", flush=True)
        return False
    if result.returncode == 0:
        print(f"OK: {repo_id}/{filename}", flush=True)
        return True
    print(f"FAIL: {repo_id}/{filename} | returncode={result.returncode}", flush=True)
    return False


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
            env=hf_download_env(),
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
    parser.add_argument(
        "--group",
        choices=["medium", "large", "all", "sd15_styles", "reference", "webapp"],
        default="medium",
        help=(
            "webapp: 推荐一键（medium+large+reference，含 Annotators 与 IP-Adapter 单文件）；"
            "sd15_styles: AOM3A3、DreamShaper、水彩/素描 LoRA 等；"
            "reference: 仅参考模式相关快照与 IP-Adapter 文件"
        ),
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or None
    print("HF_TOKEN exists:", bool(token), flush=True)

    models: list[str] = []
    if args.group == "webapp":
        models = webapp_model_ids()
    elif args.group in ("medium", "all"):
        models.extend(MEDIUM_MODELS)
    if args.group in ("large", "all"):
        for m in LARGE_MODELS:
            if m not in models:
                models.append(m)
    if args.group in ("sd15_styles", "all"):
        for m in SD15_STYLE_MODELS:
            if m not in models:
                models.append(m)
    if args.group in ("reference", "all"):
        for m in REFERENCE_SNAPSHOTS:
            if m not in models:
                models.append(m)

    ok = 0
    total = len(models)
    for m in models:
        if run_one(args.python, m, args.timeout, token):
            ok += 1

    if args.group in ("sd15_styles", "all"):
        for repo_id, fname in SD15_SINGLE_FILES:
            total += 1
            if run_one_file(args.python, repo_id, fname, args.timeout, token):
                ok += 1

    if args.group in ("reference", "all", "webapp"):
        for repo_id, fname in REFERENCE_SINGLE_FILES:
            total += 1
            if run_one_file(args.python, repo_id, fname, args.timeout, token):
                ok += 1

    print(f"\nSummary: {ok}/{total} success", flush=True)
    return 0 if ok == total else 2


if __name__ == "__main__":
    sys.exit(main())
