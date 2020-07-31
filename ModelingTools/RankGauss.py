# Reference:
# 1. rankGauss by Micheal Jahrer
# https://github.com/michaeljahrer/rankGauss/blob/master/rankGaussMain.cpp
# 2. a beginning of scikit-learn compatible implementation of GaussRank 
# https://github.com/zygmuntz/gaussrank/blob/master/gaussrank.py

# Extended by 오병화, 20180903

from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn.utils import check_array, check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import scale
from scipy.interpolate import interp1d
from collections import OrderedDict
from itertools import chain
#from numba import jit
import numpy as np
import pandas as pd
import pickle
import json

try:
    from joblib import Parallel, delayed
except ImportError:
    from sklearn.externals.joblib import Parallel, delayed

# nan_to_val : finite 하지 않은 값(np.nan, np.inf, -np.inf)이 데이터에 있을 경우, 어떻게 처리할지를 결정한다. (기본값 None)
### None / True - Finite 하지 않은 값들을 따로 빼놓고 RankGauss Scaling을 수행한 후, 이 값들을 0 으로 Assign
### 숫자값 - True 일 때와 동일하게 작동하나, 나중에 Finite 하지 않은 값들을 특정 숫자값으로 Assign
### False - Finite 하지 않은 값이 데이터에 있을 경우 에러 메시지를 출력하여 종료

# extrapolate : 기본값 False
### False - 앞서 fit() 호출 때 입력된 값들에서의 Min, Max 값을 저장해 놓았다가 transform() 함수가 호출됨
### True - 외삽법을 사용하여 값을 추천

# num_storing : 이 값은 fit() 함수가 호출되었을 때, 변수별로 내부적으로 저장할 값들 (X ~ Transformed Codebooks) 의 갯수를 설정한다.
# 이 codebook은, 테스트 데이터에 대해 이미 fit 된 정보에 따라 적절한 값을 Assign하기 위해 필요하다. 이 값이 클수록 메모리를 많이 요구한다. (기본값 None)
### None - 저장하는 값들의 수에 제약을 두지 않는다. (Maximum 64-bit integer Number 로 세팅)
### 숫자값 - 각 변수별로 숫자 갯수의 X ~ Transformed Values Pair 를 저장한다.

# random_state : 일반적인 sklearn 에서의 그 Random State와 같다. 
# 위 num_storing 이 전체 데이터 값 수보다 작을 때 샘플링을 해야 하는데 그 때 사용된다. (기본값 None)

# interp_params : transform(), inverse_transform() 함수가 호출될 경우 새 데이터에 대해서 값을 유추하는데, 
# 이 때 interpolation 기법을 사용한다. 구체적으로, scipy.interpolate.interp1d 를 사용하는데, 이 함수의 동작을 추가로 제어하는
# 매개변수를 dict 형태로 줄 수 있다.

# n_jobs: 이 값은 작업을 수행될 때 몇 개의 Process를 동시에 사용하여 작업을 수행할지 결정한다.
# 데이터의 각 컬럼(변수)별로 수행되는 작업을 병렬화하기에, n_jobs를 변수 갯수 이상 지정하는 것은 의미가 없다. (기본값 None)
### None - sklearn.externals.joblib.Parallel 의 기본값에 따라 1 또는 all cores
### 숫자값 - Process 를 숫자값만큼 사용한다.
### -1 - 시스템에서 사용 가능한 모든 Process를 사용한다. 

class RankGaussScaler(BaseEstimator, TransformerMixin):
    def __init__(self, nan_to_val=None, extrapolate=False, num_storing=None, random_state=None, interp_params=None, n_jobs=None):
        nan_to_val = nan_to_val or True
        self.nan_to_val = 0.0 if isinstance(nan_to_val, bool) else nan_to_val
        self.force_all_finite = False
        if isinstance(nan_to_val, bool) and not nan_to_val:
            self.force_all_finite = True
        self.extrapolate = extrapolate
        num_storing = num_storing or np.iinfo(int).max
        self.num_storing = 2 if num_storing < 2 else num_storing
        self.random_state = check_random_state(random_state)
        self.interp_params = interp_params or dict(kind='linear')
        self.n_jobs = n_jobs
    
    def fit(self, X, y=None):
        X = self._check_array(X)
        X = self._to_2d_if_1d(X)
        self.codebooks_ = Parallel(n_jobs=self.n_jobs)(delayed(self._make_codebook)(*x) for x in enumerate(X.T))

        return self
    
    def transform(self, X):
        transformed = self._transform(X, self._transform_column)
        if not self.force_all_finite:
            transformed[~np.isfinite(transformed)] = self.nan_to_val
        
        return transformed
    
    def inverse_transform(self, X):
        return self._transform(X, self._inv_transform_column)
    
    def _transform(self, X, func_transform):
        X = self._check_before_transform(X)
        return_as_1d = True if len(X.shape) == 1 else False
        X = self._to_2d_if_1d(X)
    
        transformed = np.array(Parallel(n_jobs=self.n_jobs)(delayed(func_transform)(*x, **self.interp_params) for x in enumerate(X.T))).T
    
        return self._to_1d_if_single(transformed) if return_as_1d else transformed
  
    def _check_array(self, X):
        # validate input and return X as numpy format
        return check_array(X, dtype=FLOAT_DTYPES, ensure_2d=False, force_all_finite=self.force_all_finite)
  
    def _check_num_cols(self, X):
        # validate input after fit()
        num_features = 1 if len(X.shape) == 1 else X.shape[1]
        if num_features != len(self.codebooks_):
            raise ValueError('bad input shape {0}'.format(X.shape))
      
    def _check_before_transform(self, X):
        check_is_fitted(self, 'codebooks_') # check if 'codebooks_'
        X = self._check_array(X)                # check input type and struncture
        self._check_num_cols(X)                # check # of columns
        return X
  
    def _make_codebook(self, col_index, x):
        codebook = build_rankguass_trafo(x)
        num_codes = len(codebook[0])
    
        if num_codes == 0:
            raise ValueError('column %d contains only null values' % col_index)
        elif num_codes > self.num_storing:
            # first, select minimum and maxmum, then choose the rest randomly
            chosen = self.random_state.choice(num_codes - 2, self.num_storing - 2, replace=False)
            return codebook[0][chosen], codebook[1][chosen]
        else:
            return codebook
    
    def _transform_column(self, index, x, **interp1d_params):
        return self._transform_with_interp(x, *self.codebooks_[index], **interp1d_params)

    def _inv_transform_column(self, index, x, **inter1d_params):
        return self._transform_with_interp(x, *reversed(self.codebooks_[index]), **interp1d_params)

    def _transform_with_interp(self, x, train_x, train_y, **interp1d_params):
        if len(train_x) == 1:
            return np.ones(x.shape) * train_y[0]
        f = interp1d(train_x, train_y, fill_value='extrapolate', **interp1d_params)
        return f(x) if self.extrapolate else f(np.clip(x, *minmax(train_x)))

    @staticmethod
    def _to_2d_if_1d(a):
        return a.reshape(-1, 1) if len(a.shape) == 1 else a

    @staticmethod
    def _to_1d_if_single(a):
        return a.ravel() if a.shape[1] == 1 else a
    

# function for simultaneous max() and min() (using numba)
# https://stackoverflow.com/a/33919126
#@jit
def minmax(x):
    maximum = x[0]
    minimum = x[0]
    for i in x[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i

    return minimum, maximum

# converted from [ref 1]
#@jit
def norm_cdf_inv(p):
    sign = 1.0
    if p < 0.5:
        sign = -1.0
    else:
        p = 1.0 - p
    t = np.sqrt(-2.0 * np.log(p))
    return sign * (t - ((0.010328 * t + 0.802853) * t + 2.515517)  / (((0.001308 * t + 0.189269) * t + 1.432788) * t + 1.0))

# converted from [ref 1]
#@jit
def build_rankguass_trafo(x):
    finite_indices = np.isfinite(x)
    if np.sum(finite_indices) == 0:
        return np.array([]), np.array([])
    x_finite = x[np.isfinite(x)]

    hist = dict()
    for val in x_finite:
        hist[val] = hist.get(val, 0) + 1

    len_hist = len(hist)
    list_keys = list(hist.keys())

    if len_hist == 1:
        return np.array(list_keys), np.array([0.0])
    elif len_hist == 2:
        return np.array(list_keys), np.array([0.0, 1.0])
    else:
        hist = OrderedDict(sorted(hist.items())) # sort by key
        n = float(x_finite.shape[0])
        cnt = 0.0
        mean = 0.0
        trafo_keys = list()
        trafo_values = list()

        for key, val in hist.items():
            # (notice) 'cnt / n * 0.98 + 1e-3' is always larger than zero
            rank_v = norm_cdf_inv(cnt / n * 0.998 + 1e-3) * 0.7
            trafo_keys.append(key)
            trafo_values.append(rank_v)
            mean += val * rank_v
            cnt += val

        mean /= n
        trafo_values = np.array(trafo_values)
        trafo_values -= mean

        return np.array(trafo_keys), trafo_values