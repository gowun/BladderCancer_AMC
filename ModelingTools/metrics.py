import pandas as pd
import numpy as np
import sklearn.metrics as met
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import graphviz
from scipy.stats import ttest_ind, ttest_rel


def compute_performance(real_y, pred_y, prob):
    performance = dict()
    performance['AUC'] = met.roc_auc_score(real_y, prob)
    performance['accuracy'] = met.accuracy_score(real_y, pred_y)
    performance['precision'] = met.precision_score(real_y, pred_y)
    performance['recall'] = met.recall_score(real_y, pred_y)
    performance['f1'] = met.f1_score(real_y, pred_y)
    performance['mcc'] = met.matthews_corrcoef(real_y, pred_y)
    performance['R2'] = met.r2_score(real_y, prob)
    
    print(performance)

    return performance


def draw_auc(y_true, y_score_list, pos_label=1, label_list=None):
  sns.set(palette = "colorblind", font_scale = 1.35, rc = {"figure.figsize": (8, 6), "axes.facecolor": ".92"})
  all_points = list(map(lambda x: met.roc_curve(y_true, x, pos_label=pos_label), y_score_list))

  plt.figure()
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.plot([0, 1], [0, 1], color='black', linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  for i, p in enumerate(all_points):
    c = np.random.rand(3)
    l = "AUC = %0.2f" % (met.auc(p[0], p[1]))
    if label_list is not None:
      l = label_list[i] + ' (' + l + ')'
    plt.plot(p[0], p[1], color=c, lw=2, label=l)
  plt.legend(loc="lower right")
  plt.show()


# mode in ['ind', 'rel']
def do_ttest_2dfs(df_list, min_pvalue, mode='ind'):
  cols = df_list[0].columns

  


def compute_shap_value(model, X):
    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(X)[1]

    return shap_values


def each_decision_plot(ith, X, shap_values, y_ratio):
    shap.decision_plot(y_ratio, shap_values[ith], X.iloc[ith], link='logit')


def each_force_plot(ith, X, shap_values, y_ratio):
    shap.force_plot(y_ratio, shap_values[ith], X.iloc[ith], link='logit', matplotlib=True)