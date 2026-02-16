"""JVNVコーパスデータローダ。

ディレクトリ構造: {speaker}/{emotion}/{session}/{speaker}_{emotion}_{session}_{id}.wav
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class JVNVSample:
    """JVNVコーパスの1音声サンプル。"""

    audio_path: str
    emotion: str
    speaker: str
    session: str


def load_jvnv(root_dir: str | Path) -> list[JVNVSample]:
    """JVNVコーパスのWAVファイルを走査し全サンプルをリスト化する。

    Args:
        root_dir: JVNVコーパスのルートディレクトリ (例: data/jvnv_v1)

    Returns:
        JVNVSampleのリスト

    Raises:
        FileNotFoundError: root_dirが存在しない場合

    """
    root = Path(root_dir)
    if not root.exists():
        msg = f"JVNV root directory not found: {root}"
        raise FileNotFoundError(msg)

    samples: list[JVNVSample] = []
    # {speaker}/{emotion}/{session}/*.wav
    for wav_path in sorted(root.glob("*/*/*/**.wav")):
        parts = wav_path.relative_to(root).parts
        if len(parts) < 4:
            continue
        speaker, emotion, session = parts[0], parts[1], parts[2]
        samples.append(
            JVNVSample(
                audio_path=str(wav_path),
                emotion=emotion,
                speaker=speaker,
                session=session,
            ),
        )

    return samples
