import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from .metrics import *
from .survival_analysis import *

def logiReg_model_with_performance(train_X_y, max_iter, valid_X_y=None, class_weight=None):
    X, y = train_X_y
    clf = LogisticRegression(class_weight=class_weight, max_iter=max_iter).fit(X, y)
    if valid_X_y is None:
        vX = X
        vy = y
    else:
        vX, vy = valid_X_y
    pred = clf.predict(vX)
    prob = clf.predict_proba(vX)[:, 1]
    
    performance = compute_performance(vy, pred, prob)

    result = dict()
    result['model'] = clf
    result['coefficients'] = pd.DataFrame({'feature': [''] + list(X.columns), 'coefficient': list(clf.intercept_) + list(clf.coef_[0])})
    result['performance'] = performance
    result['columns'] = X.columns

    return result


def train_and_filter_models(train_X_y, col_list, max_iter, min_auc, event, duration, max_pvalue, valid_X=None):
    X, y = train_X_y
    if valid_X is None:
        vX = X
    else:
        vX = valid_X
    X_list = list(map(lambda x: X[x], col_list))
    mds = list(map(lambda x: logiReg_model_with_performance([x, y], max_iter, class_weight='balanced'), X_list))
    mds = list(filter(lambda x: x['performance']['AUC'] >= min_auc, mds))
    preds = list(map(lambda x: x['model'].predict(vX[x['columns']]), mds))
    ps = list(map(lambda x: logrank_pvalue(duration, x, event), preds))
    mds = list(filter(lambda x: x[1] <= max_pvalue, zip(mds, ps)))
    mds = list(map(lambda x: x[0], mds))

    tmp = list(map(lambda x: x['performance']['AUC'], mds))
    order = list(zip(range(len(mds)), tmp))
    order = sorted(order, key=lambda a: a[1], reverse=True)
    models = list()
    for o, _ in order:
        models.append(mds[o])
    
    return models
