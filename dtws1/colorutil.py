from colormath.color_objects import sRGBColor, HSVColor
from colormath.color_conversions import convert_color
import numpy as np


class ColorSelectorBase:
    @staticmethod
    def _make_cols(n, hue):
        return [
            convert_color(
                HSVColor(hue, i/(n-1), 1),
                sRGBColor
            ).get_rgb_hex()
            for i in range(n)
        ]

    def __call__(self, v):
        pos = 0
        while pos < self._n_tick and self._tick[pos] <= v:
            pos += 1
        return self._cols[pos]


class color_selector(ColorSelectorBase):
    def __init__(self, values, n, hue=0.9, min_value=-np.Inf,
                 max_value=np.Inf, desc=False):
        self._n_cols = n
        self._n_tick = n - 1
        self._cols = ColorSelectorBase._make_cols(n, hue)
        if desc:
            self._cols = list(reversed(self._cols))
        m = max(min_value, min(values))
        M = min(max_value, max(values))
        self._tick = np.linspace(m, M, self._n_cols+1)[1:-1]


class color_selector_tick(ColorSelectorBase):
    def __init__(self, ticks, hue=0.9, reverse=False):
        self._n_cols = len(ticks) + 1
        self._n_tick = len(ticks)
        self._cols = ColorSelectorBase._make_cols(self._n_cols, hue)
        if reverse:
            self._cols = list(reversed(self._cols))
        self._tick = ticks


class color_selector_p(ColorSelectorBase):
    def __init__(self, values, n, hue=0.9):
        self._n_tick = n - 1
        self._n_cols = n
        self._cols = ColorSelectorBase._make_cols(self._n_cols, hue)
        ps = np.linspace(0, 100, self._n_cols+1)[1:-1]
        self._tick = np.percentile(values, ps)
