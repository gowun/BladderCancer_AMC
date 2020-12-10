import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
import matplotlib.pyplot as plt
import seaborn as sns


def logrank_pvalue(duration, group, event):
    tmp = multivariate_logrank_test(duration, group, event)
    return tmp.p_value

def do_KM_analysis(durations, groups, events, group_labels, xlabel=None):
    fitters = list()
    ax_list = list()
    sns.set(palette = "colorblind", font_scale = 1.35, rc = {"figure.figsize": (8, 6), "axes.facecolor": ".92"})
    
    for i, cl in enumerate(sorted(set(groups))):
        kmf = KaplanMeierFitter()
        kmf.fit(durations[groups == cl], events[groups == cl], label=group_labels[i])
        fitters.append(kmf)
        if i == 0:
            ax_list.append(kmf.plot(ci_show=False))
        elif i == len(group_labels)-1:
            kmf.plot(ax=ax_list[-1], ci_show=False)
        else:
            ax_list.append(kmf.plot(ax=ax_list[-1], ci_show=False))
        
    add_at_risk_counts(*fitters, labels=group_labels)
    ax_list[-1].set_ylim(0,1.1)
    if xlabel is not None:
        ax_list[-1].set_xlabel(xlabel)

    multi = multivariate_logrank_test(durations, groups, events)
    ax_list[-1].text(0.1, 0.01, 'P-value=%.3f'% multi.p_value)
    
    if len(set(groups)) > 2:
      pair = pairwise_logrank_test(durations, groups, events)
      pair.print_summary()
    
    plt.show()
    
    return kmf