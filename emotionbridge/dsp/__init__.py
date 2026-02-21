from typing import TYPE_CHECKING

from .mapper import EmotionDSPMapper
from .types import DSPControlVector

if TYPE_CHECKING:
    from .processor import EmotionDSPProcessor

__all__ = [
    "DSPControlVector",
    "EmotionDSPMapper",
    "EmotionDSPProcessor",
]


def __getattr__(name: str):
    if name == "EmotionDSPProcessor":
        from .processor import EmotionDSPProcessor

        return EmotionDSPProcessor
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
