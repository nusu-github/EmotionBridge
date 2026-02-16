"""VOICEVOX Engine REST APIの非同期クライアント。"""

import asyncio
import logging
from types import TracebackType
from typing import Any, Self

import httpx

from emotionbridge.tts.exceptions import (
    VoicevoxAPIError,
    VoicevoxConnectionError,
    VoicevoxTimeoutError,
)
from emotionbridge.tts.types import AudioQuery, SpeakerInfo, SpeakerStyle

logger = logging.getLogger(__name__)


class VoicevoxClient:
    """VOICEVOX Engine REST APIの非同期クライアント。

    httpxベースの非同期HTTPクライアントで、リトライ・タイムアウト・
    例外変換を提供する。コンテキストマネージャとして使用可能。

    使用例::

        async with VoicevoxClient() as client:
            query = await client.audio_query("こんにちは", speaker_id=0)
            wav = await client.synthesis(query, speaker_id=0)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:50021",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ) -> None:
        self._base_url = base_url
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    # ------------------------------------------------------------------
    # 内部: リトライ付きリクエスト
    # ------------------------------------------------------------------

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """指数バックオフ付きリトライでHTTPリクエストを実行する。

        4xxクライアントエラーはリトライせず即座にVoicevoxAPIErrorを送出する。
        5xxサーバーエラー・タイムアウト・接続エラーはリトライ対象。
        """
        last_exception: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.TimeoutException as e:
                last_exception = VoicevoxTimeoutError(str(e))
                logger.warning(
                    "VOICEVOX API timeout (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries,
                    e,
                )
            except httpx.ConnectError as e:
                last_exception = VoicevoxConnectionError(str(e))
                logger.warning(
                    "VOICEVOX connection error (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries,
                    e,
                )
            except httpx.HTTPStatusError as e:
                # 4xx はリトライしない（クライアントエラー）
                if 400 <= e.response.status_code < 500:
                    raise VoicevoxAPIError(
                        e.response.status_code,
                        e.response.text,
                    ) from e
                last_exception = VoicevoxAPIError(
                    e.response.status_code,
                    e.response.text,
                )
                logger.warning(
                    "VOICEVOX API error (attempt %d/%d): %s %s",
                    attempt + 1,
                    self._max_retries,
                    e.response.status_code,
                    e.response.text,
                )

            if attempt < self._max_retries - 1:
                delay = self._retry_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise last_exception  # type: ignore[misc]

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """GET /version でエンジンの疎通を確認する。

        Returns:
            True: エンジンが応答した。
            False: 接続失敗またはタイムアウト。

        """
        try:
            response = await self._client.get("/version")
            response.raise_for_status()
        except (httpx.HTTPError, httpx.StreamError):
            return False
        else:
            return True

    async def speakers(self) -> list[SpeakerInfo]:
        """GET /speakers で利用可能なキャラクター一覧を取得する。

        Returns:
            SpeakerInfoのリスト。
        Raises:
            VoicevoxAPIError: API呼び出し失敗時。
            VoicevoxConnectionError: 接続失敗時。
            VoicevoxTimeoutError: タイムアウト時。

        """
        response = await self._request_with_retry("GET", "/speakers")
        return [
            SpeakerInfo(
                name=s["name"],
                speaker_uuid=s["speaker_uuid"],
                styles=[
                    SpeakerStyle(
                        name=st["name"],
                        id=st["id"],
                        style_type=st.get("type", "talk"),
                    )
                    for st in s["styles"]
                ],
                version=s.get("version", ""),
            )
            for s in response.json()
        ]

    async def audio_query(self, text: str, speaker_id: int) -> AudioQuery:
        """POST /audio_query でテキストから音声合成クエリを生成する。

        Args:
            text: 合成対象のテキスト。
            speaker_id: スピーカースタイルID。
        Returns:
            AudioQuery。
        Raises:
            VoicevoxAPIError: API呼び出し失敗時。
            VoicevoxConnectionError: 接続失敗時。
            VoicevoxTimeoutError: タイムアウト時。

        """
        response = await self._request_with_retry(
            "POST",
            "/audio_query",
            params={"text": text, "speaker": speaker_id},
        )
        return AudioQuery.from_dict(response.json())

    async def synthesis(self, audio_query: AudioQuery, speaker_id: int) -> bytes:
        """POST /synthesis でAudioQueryから音声を合成する。

        Args:
            audio_query: 音声合成クエリ。
            speaker_id: スピーカースタイルID。
        Returns:
            WAV形式の音声バイナリ。
        Raises:
            VoicevoxAPIError: API呼び出し失敗時。
            VoicevoxConnectionError: 接続失敗時。
            VoicevoxTimeoutError: タイムアウト時。

        """
        response = await self._request_with_retry(
            "POST",
            "/synthesis",
            params={"speaker": speaker_id},
            json=audio_query.to_dict(),
        )
        logger.debug("VOICEVOX synthesis completed: %d bytes", len(response.content))
        return response.content

    async def initialize_speaker(
        self,
        speaker_id: int,
        skip_reinit: bool = True,
    ) -> None:
        """POST /initialize_speaker でスタイルを事前初期化する。

        Args:
            speaker_id: スピーカースタイルID。
            skip_reinit: 初期化済みスタイルの再初期化をスキップ。
        Raises:
            VoicevoxAPIError: API呼び出し失敗時。
            VoicevoxConnectionError: 接続失敗時。
            VoicevoxTimeoutError: タイムアウト時。

        """
        await self._request_with_retry(
            "POST",
            "/initialize_speaker",
            params={"speaker": speaker_id, "skip_reinit": skip_reinit},
        )

    # ------------------------------------------------------------------
    # ライフサイクル
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """HTTPクライアントを閉じる。"""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
