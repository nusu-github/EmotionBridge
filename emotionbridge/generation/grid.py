"""パラメータグリッドサンプラー。

5D制御パラメータ空間 [-1, +1]^5 からサンプルを生成する。
Latin Hypercube Sampling (LHS) または全数探索 (Full Grid) に対応。
"""

import logging
from enum import StrEnum

import numpy as np
from scipy.stats.qmc import LatinHypercube

from emotionbridge.config import GridConfig
from emotionbridge.constants import NUM_CONTROL_PARAMS

logger = logging.getLogger(__name__)


class SamplingStrategy(StrEnum):
    """サンプリング戦略。"""

    LHS = "lhs"
    FULL_GRID = "full_grid"


class GridSampler:
    """5D制御パラメータ空間のサンプラー。

    LHSモードでは各テキストに対して決定論的なシードを導出し、
    再現可能なサンプリングを行う。Full Gridモードでは各軸を等間隔に
    離散化した直積を返す。
    """

    def __init__(self, config: GridConfig) -> None:
        self._config = config
        self._strategy = SamplingStrategy(config.strategy)

    def sample(self, text_id: int) -> np.ndarray:
        """指定テキストに対する制御パラメータ群を生成する。

        Args:
            text_id: テキストの一意識別子（LHSシード導出に使用）。

        Returns:
            shape (n_samples, 5) の ndarray。各値は [-1.0, +1.0]。

        """
        if self._strategy == SamplingStrategy.LHS:
            seed = self._config.random_seed ^ text_id
            return self.sample_lhs(self._config.lhs_samples_per_text, seed)
        return self.sample_full_grid()

    def sample_lhs(self, n_samples: int, seed: int) -> np.ndarray:
        """Latin Hypercube Samplingで n_samples 点を5D空間から生成する。

        scipy.stats.qmc.LatinHypercubeを使用し、[0, 1]^5 の空間を
        [-1, +1]^5 にスケーリングする。

        Args:
            n_samples: サンプル数。
            seed: 乱数シード。

        Returns:
            shape (n_samples, 5) の ndarray。各値は [-1.0, +1.0]。

        """
        sampler = LatinHypercube(d=NUM_CONTROL_PARAMS, seed=seed)
        # [0, 1]^5 のサンプルを生成
        unit_samples = sampler.random(n=n_samples)
        # [-1, +1]^5 にスケール
        scaled = unit_samples * 2.0 - 1.0
        return scaled.astype(np.float32)

    def sample_full_grid(self) -> np.ndarray:
        """全数探索グリッドを生成する。

        各軸を config.grid_steps 段階で [-1, +1] に等間隔離散化し、
        その直積を返す。

        Returns:
            shape (grid_steps^5, 5) の ndarray。各値は [-1.0, +1.0]。

        """
        steps = self._config.grid_steps
        axis = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
        grids = np.meshgrid(*([axis] * NUM_CONTROL_PARAMS), indexing="ij")
        # (steps^5, 5) に整形
        return np.column_stack([g.ravel() for g in grids])

    @property
    def total_samples_per_text(self) -> int:
        """テキスト1件あたりのパラメータ組み合わせ数。"""
        if self._strategy == SamplingStrategy.LHS:
            return self._config.lhs_samples_per_text
        return self._config.grid_steps**NUM_CONTROL_PARAMS
