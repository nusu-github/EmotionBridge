"""VOICEVOX Engine関連の例外階層。"""


class VoicevoxError(Exception):
    """VOICEVOX関連エラーの基底クラス。"""


class VoicevoxConnectionError(VoicevoxError):
    """VOICEVOX Engineへの接続失敗。"""


class VoicevoxAPIError(VoicevoxError):
    """VOICEVOX APIがエラーレスポンスを返した。"""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"VOICEVOX API error {status_code}: {detail}")


class VoicevoxTimeoutError(VoicevoxError):
    """VOICEVOX APIリクエストがタイムアウトした。"""
