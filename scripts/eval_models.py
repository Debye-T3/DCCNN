import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Evaluate multiple models on fixed H5 files.")
    parser.add_argument("--models", nargs="+", required=True, help="List of model .pt paths.")
    parser.add_argument("--input-files", nargs="+", required=True, help="Fixed H5 files for evaluation.")
    parser.add_argument("--config", default="config/config_baseline.yaml", help="Path to config YAML.")
    parser.add_argument("--output-dir", default="results_ms_txt/eval", help="Directory for outputs.")
    parser.add_argument("--target-key", default="spectrum", help="Target dataset key.")
    return parser.parse_args()


def build_glob(files: list[str]) -> str:
    # run_inference.py 需要 glob，这里用花括号拼成“文件列表”形式
    if len(files) == 1:
        return files[0]
    return "{" + ",".join(files) + "}"


def main() -> None:
    # 逐个模型运行推理评估
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.run_inference import run_inference

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        run_inference(
            Path(args.config).resolve(),
            Path(model).resolve(),
            "",
            output_dir,
            target_key_override=args.target_key,
            input_files=args.input_files,
        )


if __name__ == "__main__":
    main()
