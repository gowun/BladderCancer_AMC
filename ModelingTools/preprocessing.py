import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from .RankGauss import RankGaussScaler
from scipy.stats import normaltest, ttest_ind_from_stats


def scale_with_scalers(df, selected_scalers):
    X = df.astype(float)
    if selected_scalers == 'all':
        selected_scalers = ['log2', 'log2_minmax', 'standard', 'minmax', 'max_abs', 'robust', 'power', 'quantile', 'l2', 'rankgauss']
    if 'log2' in ''.join(selected_scalers):
        X_log2 = pd.DataFrame()
        for c in X.columns:
            X_log2[c] = np.log2(X[c]).apply(lambda x: max([x, 0.0]))
    scaled_dict = dict()
    for sc in selected_scalers:
        if sc == 'log2':
            scaled_dict[sc] = X_log2.values
        elif sc == 'log2_minmax': 
            scaled_dict[sc] = MinMaxScaler().fit_transform(X_log2)
        elif sc == 'standard': 
            scaled_dict[sc] = StandardScaler().fit_transform(X)
        elif sc == 'minmax': 
            scaled_dict[sc] = MinMaxScaler().fit_transform(X)
        elif sc == 'max_abs': 
            scaled_dict[sc] = MaxAbsScaler().fit_transform(X)
        elif sc == 'robust':
            scaled_dict[sc] = RobustScaler(quantile_range=(5, 95)).fit_transform(X)
        elif sc == 'power': 
            scaled_dict[sc] = PowerTransformer(method='yeo-johnson').fit_transform(X)
        elif sc == 'quantile': 
            scaled_dict[sc] = QuantileTransformer(output_distribution='normal').fit_transform(X)
        elif sc == 'l2':
            scaled_dict[sc] = Normalizer().fit_transform(X)
        elif sc == 'rankgauss':
            scaled_dict[sc] = RankGaussScaler().fit_transform(X)
        scaled_dict[sc] = pd.DataFrame(scaled_dict[sc], columns=df.columns)
    
    return scaled_dict


def compare_two_samples_and_draw_feasible_columns(sample1, sample2, min_pvalue=0.01):
    col1 = set(sample1.columns)
    col2 = set(sample2.columns)
    interset_cols = sorted(col1.intersection(col2))

    passed_cols = []
    for c in interset_cols:
        # For a different distribution, we can reject the null hypothesis since the pvalue is below 1%
        # This is a two-sided test for the null hypothesis that two independent samples have identical average (expected) values
        s1, s2 = sample1[c], sample2[c]
        t, pvalue = ttest_ind_from_stats(np.mean(s1), np.std(s1), len(s1), np.mean(s2), np.std(s2), len(s2))
        if pvalue >= min_pvalue:
            passed_cols.append(c)
    
    return passed_cols


def handle_repeated_columns(df, mode='avg'):
    cols = sorted(set(df.columns))

    frq_dict = dict()
    df_dict = dict()
    for c in cols:
        tmp = df[c].shape
        if len(tmp) == 1:
            frq_dict[c] = 1
        else:
            frq_dict[c] = tmp[1]
        if frq_dict[c] == 1:
            df_dict[c] = list(df[c])
        else:
            if mode == 'avg':
                df_dict[c] = list(df[c].mean(1))
            elif mode == 'max':
                df_dict[c] = list(df[c].max(1))
            elif mode == 'min':
                df_dict[c] = list(df[c].min(1))
    
    return frq_dict, pd.DataFrame(df_dict, index=df.index)

