import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from itertools import combinations
from multiprocessing import Pool
from .utils import *


sns.set(color_codes=True)


def max_len(letter, split_str):
  tfs = np.array(split_str) == letter
  len_list = []
  ln = 0
  for i in tfs:
    if i:
      ln += 1
    else:
      len_list.append(ln)
      ln = 0
  len_list.append(ln)
  return max(len_list)


def find_max_len_by_ch(string, ch_list, splt='_'):
  split_str = string.split(splt)
  return list(map(lambda x: max_len(x, split_str), ch_list))


def compute_linkage_impurity(link_info, class_labels):
  def find_values(LRmode, which, labels):
    v = int(link_info.iloc[which][LRmode])
    if v in range(len(class_labels)):
      labels[LRmode][which] = class_labels[v]
    else:
      which_ = list(link_info['new_level']).index(v)
      labels[LRmode][which] = '_'.join([labels['left'][which_], labels['right'][which_]])
    return labels
  
  class_labels = list(map(lambda x: str(x), class_labels))
  link_info['new_level'] = list(range(len(class_labels), len(class_labels) + len(link_info)))
  labels = dict()
  labels['left'] = [''] * len(link_info)
  labels['right'] = [''] * len(link_info)

  done = np.array([False] * len(link_info))
  level = 2
  while sum(done) < len(link_info):
    idx = link_info.loc[link_info['level'] == level].index
    for i in idx:
      labels = find_values('left', i, labels)
      labels = find_values('right', i, labels)
      done[i] = True
    level += 1
  link_info['left_label'] = labels['left']
  link_info['right_label'] = labels['right']

  lb = sorted(set(class_labels))
  cnt_lb = dict()
  targets = []
  for l in lb:
    cnt_lb[l] = sum(np.array(class_labels) == l)
    targets.append('_'.join([l] * cnt_lb[l]))
  
  str_arr = link_info[['left_label', 'right_label']].values 
  for ss in str_arr:
    in_tfs = list(map(lambda s: list(map(lambda x: x in s, targets)), ss))
    in_tfs = in_tfs[0] + in_tfs[1]
    if True in in_tfs:
      break
  try:
    which = in_tfs.index(True)
    if which < len(targets):
      tt = targets[which]
      s = ss[0]
    else:
      tt = targets[which - len(targets)]
      s = ss[1]
    if tt == s:
      impurity = 0
    else:
      l1 = len(s.split('_'))
      l2 = len(tt.split('_'))
      impurity = l1 - l2
  except ValueError:
    impurity = len(class_labels)
  
  return impurity, link_info


def compute_cluster_impurity(df, class_labels, method='average', metric='euclidean'):
    links = pd.DataFrame(linkage(df.values, method=method, metric=metric), columns=['left', 'right', 'distance', 'level'])
    return compute_linkage_impurity(links, class_labels)


def find_features_of_lowest_impurity(feature_sets, df, class_labels):
  impurity_list = []
  for fs in feature_sets:
    im, _ = compute_cluster_impurity(df[fs], class_labels)
    impurity_list.append(im)

  result = dict()
  result['impurity'] = min(impurity_list)
  result['features'] = list(feature_sets)[impurity_list.index(result['impurity'])]
  return result


def plot_cluster_heatmap(df, transpose=False, method='average', metric='euclidean', title=None, figsize=None):
    if transpose:
        df = transpose_df(df)
    df = df.astype(float)
    sns.clustermap(df, method=method, metric=metric, figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_sequential_cluster_heatmap(df, var_list, transpose=False, title_list=None):
    for i, vs in enumerate(var_list):
        ins = list(map(lambda x: x in df.columns, vs))
        var = np.array(vs)[ins]
        dd = {'df': df[var], 'transpose': transpose}
        if title_list is not None:
            dd['title'] = title_list[i]
        plot_cluster_heatmap(**dd)


def parallel_impurity(arg_set):
  fts_set, df, labels = arg_set
  rr = find_features_of_lowest_impurity(fts_set, df, labels)
  return rr


def find_features_of_lowest_impurity_parallel(nProc, feature_comb, df, labels):
  block_size = round(len(feature_comb) / nProc)
  print(block_size)
  block_start_idx = list(range(0, len(feature_comb), block_size))
  blocks = list()
  feature_comb = np.array(feature_comb)
  for i, s in enumerate(block_start_idx):
    if i < len(block_start_idx) - 1:
      blocks.append(feature_comb[s:block_start_idx[i+1]])
    else:
      blocks.append(feature_comb[s:])
  pool = Pool(processes=nProc)
  blocks = list(map(lambda x: [x, df, labels], blocks))
  result = pool.map(parallel_impurity, blocks)
  im_list = list(map(lambda x: x['impurity'], result))
  
  min_im = min(im_list)
  min_idx = im_list.index(min_im)
  return result[min_idx]


def find_best_feature_comb_parallel(X, y, varlist_dict, nProc=20, nMin=5, save_path=None):
    best_comb = dict()
    for i, vv in enumerate(varlist_dict.items()):
        print('~~~' + str(i) + '~~~')
        k, vs = vv
        min_impurity = find_features_of_lowest_impurity([vs], X, y)
        if min_impurity['impurity'] == 0 or len(vs) <= nMin:
            best_comb[k] = vs
        else:
            for n_fs in range(nMin, len(vs)):
                comb = list(map(lambda x: list(x), list(combinations(vs, n_fs))))
                print(len(comb))
                if len(comb) > nProc:
                    result = find_features_of_lowest_impurity_parallel(nProc, comb, X, y)
                else:
                    result = find_features_of_lowest_impurity_parallel(len(comb), comb, X, y)

                if result['impurity'] == 0:
                    break
                elif result['impurity'] < min_impurity['impurity'] or (len(result['features']) < len(min_impurity['features']) and result['impurity'] == min_impurity['impurity']):
                    min_impurity = result
            best_comb[k] = result['features']
        print([k, min_impurity])
        if save_path is not None:
            save_as_file_colab(best_comb[k], save_path + k + '.pkl', 'pickle')
    return best_comb