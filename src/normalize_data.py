import numpy as np


class MinMaxCustomScaler:
    def __init__(self):
        self._min = None
        self._max = None
        self._is_fitted = False

    def fit(self, min_value, max_value, preserve_zero=False):
        self._min = min_value
        self._max = max_value

        if preserve_zero:
            self._scale_value_func = lambda v: v / self._max if v >= 0 else -v / self._min
            self._unscale_value_func = lambda v: v * self._max if v >= 0 else -v * self._min
        else:
            self._scale_value_func = lambda v: (v - self._min) / (self._max - self._min)
            self._unscale_value_func = lambda v: v * (self._max - self._min) + self._min

        self._is_fitted = True
    
    def transform(self, x):
        if self._is_fitted:
            x_scaled = np.array([self._scale_value_func(v) for v in x])
            return x_scaled
        return None

    def inverse_transform(self, x_scaled):
        if self._is_fitted:
            x = np.array([self._unscale_value_func(v) for v in x_scaled])
            return x
        return None


class ScalersDict:
    def __init__(self):
        self.scalers_dict = {}

    def fit(self, key, min_value, max_value, preserve_zero=False):
        if key not in self.scalers_dict:
            self.scalers_dict[key] = MinMaxCustomScaler()
            self.scalers_dict[key].fit(min_value, max_value, preserve_zero)

    def transform(self, key, x):
        if key in self.scalers_dict:
            return self.scalers_dict[key].transform(x)
        return None

    def inverse_transform(self, key, x):
        if key in self.scalers_dict:
            return self.scalers_dict[key].inverse_transform(x)
        return None

# from prepare_data.stats | todo fill with train_stats_summary.csv
SCALERS_DICT = ScalersDict()
SCALERS_DICT.fit("0_acc", -600.0, 280.0, preserve_zero=True)
SCALERS_DICT.fit("1_acc", -6000.0, 29289, preserve_zero=True)
SCALERS_DICT.fit("0_steering", -460.0, 460.0, preserve_zero=True)
SCALERS_DICT.fit("1_steering", -402.0, 402.0, preserve_zero=True)
SCALERS_DICT.fit("speed", 0, 55)
# SCALERS_DICT.fit("speed", 0, 28)
