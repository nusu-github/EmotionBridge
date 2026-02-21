import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ローカル成果物ディレクトリを Hugging Face Hub へアップロードする",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="アップロード先リポジトリID（例: username/emotionbridge-generator）",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="アップロード対象のローカルディレクトリ",
    )
    parser.add_argument(
        "--path-in-repo",
        default="",
        help="リポジトリ内の配置先パス（空の場合はルート）",
    )
    parser.add_argument(
        "--repo-type",
        choices=["model", "dataset", "space"],
        default="model",
        help="Hub リポジトリ種別",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="リポジトリを private で作成する",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="アップロード先 revision（未指定時はデフォルトブランチ）",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload artifacts",
        help="コミットメッセージ",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        msg = f"local directory not found: {local_dir}"
        raise FileNotFoundError(msg)
    if not local_dir.is_dir():
        msg = f"local path must be a directory: {local_dir}"
        raise ValueError(msg)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    commit_info = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
    )

    payload = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "local_dir": str(local_dir),
        "path_in_repo": args.path_in_repo,
        "revision": args.revision,
        "commit": str(commit_info),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
