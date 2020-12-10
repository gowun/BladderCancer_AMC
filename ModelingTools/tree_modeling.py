import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import _tree
import graphviz
from sklearn.ensemble import RandomForestClassifier
from lifelines.statistics import multivariate_logrank_test
from .metrics import *
from .survival_analysis import *


def tree_model_with_performance(train_X_y, min_sample_leaf, max_depth=None, valid_X_y=None, class_weight=None):
    X, y = train_X_y
    clf = tree.DecisionTreeClassifier(class_weight=class_weight, min_samples_split=round(min_sample_leaf*5/3), min_samples_leaf=min_sample_leaf, max_depth=max_depth, random_state=1234).fit(X, y)
    if valid_X_y is None:
        vX = X
        vy = y
    else:
        vX, vy = valid_X_y
    pred = clf.predict(vX)
    prob = clf.predict_proba(vX)[:, 1]
    
    performance = compute_performance(vy, pred, prob)
    
    fi = pd.DataFrame({'feature': X.columns, 'importance': clf.feature_importances_}).sort_values(by='importance', ascending=False)
    fi = fi.loc[fi['importance'] > 0]
    
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns, proportion=True, class_names=['0', '1'], filled=True, rounded=True, special_characters=True)  
    graph = graphviz.Source(dot_data)
    
    result = dict()
    result['model'] = clf
    result['performance'] = performance
    result['feature importance'] = fi
    result['tree graph'] = graph
    result['columns'] = X.columns

    return result


def sequential_tree_modeling(data_list, y, min_sample_leaf, max_depth=None, valid_X_list=None, valid_y=None, class_weight=None):
    models = list()
    for i, d in enumerate(data_list):
        if valid_X_list is None and valid_y is None:
            tmp = tree_model_with_performance([d, y], min_sample_leaf, max_depth, class_weight=class_weight)
        else:
            tmp = tree_model_with_performance([d, y], min_sample_leaf, max_depth, [valid_X_list[i], valid_y], class_weight)
        models.append(tmp)
  
    return models


def tree_to_code(tree, feature_names):
    global script 
    script = list()
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    n_all = tree_.n_node_samples[0]
    
    def recurse(node, depth):
        indent = depth
        impurity = "{0:.3f}".format(tree_.impurity[node])
        values = tree_.value[node][0]
        score = float("{0:.3f}".format(values[1] / sum(values)))
        n = float("{0:.1f}".format(tree_.n_node_samples[node] / n_all * 100))
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = "{0:.2f}".format(tree_.threshold[node])
            script.append("{}, {} <= {}, {}, {}%, {}, {}".format(indent, name, threshold, impurity, n, values, score))
            recurse(tree_.children_left[node], depth + 1)
            script.append("{}, {} > {}, {}, {}%, {}, {}".format(indent, name, threshold, impurity, n, values, score))
            recurse(tree_.children_right[node], depth + 1)
        else:
            script.append("{}, {}, {}%, {}, {}".format(indent, impurity, n, values, score))
    
    recurse(0, 1)
    
    return script, [feature_name[0], "{0:.2f}".format(tree_.threshold[0])]


def tree_rule_summariser(script):
  line_lens = list(map(lambda x: len(x.split(',')), script))
  
  rules = list()
  for i, l in enumerate(line_lens):
    if l == 5:
      last = script[i].split(', ')
      rule = dict()
      rule['impurity'] = last[1]
      rule['ratio'] = last[2]
      samples = last[3].strip('[').strip(']').strip(' ')
      blank = samples.index(' ')
      rule['sample0'] = float(samples[:blank-1])
      rule['sample1'] = float(samples[blank+1:])
      rule['score'] = float(last[-1])
      rule['condition'] = list()
      j = i - 1
      current = last[0]
      while True:
        con = script[j].split(', ')
        if int(con[0]) < int(current) and len(con) == 6:
          current = con[0]
          try:
            where = con[1].index('<=')
            var = con[1][:where-1]
            direction = 'max'
          except ValueError:
            where = con[1].index('>')
            var = con[1][:where-1]
            tmp = con[1][where+1:] + ' < ' + var
            con[1] = tmp
            direction = 'min'
          in_not = np.array(list(map(lambda x: var in x, rule['condition'])))
          try:
            which = np.array(range(len(in_not)))[in_not][0]
          except IndexError:
            which = -1
          if sum(in_not) == 0:
            rule['condition'].append(con[1])
          else:
            r = rule['condition'][which]
            if direction == 'min':
              if ' < ' in r:
                if float(r[:r.index(' < ')]) < float(con[1][:con[1].index(' < ')]):
                  rr = con[1][:con[1].index(' < ')] + ' < ' + r[r.index(' < '):]
                  rule['condition'][which] = rr
              else:
                rule['condition'][which] = con[1][:con[1].index(' < ')] + ' < ' + r
            elif direction == 'max':
              if '<=' in r:
                if float(r[r.index('<=')+2:]) > float(con[1][con[1].index('<=')+2:]):
                  rr = r[:r.index('<=')-1] + ' <= ' + con[1][con[1].index('<=')+2:]
                  rule['condition'][which] = rr
              else:
                rule['condition'][which] = r + ' <= ' + con[1][con[1].index('<=')+2:]
        if int(current) == 1:
          break
        j -= 1
      rule['condition'] = ', '.join(rule['condition'])
      rules.append(rule)
    
  return pd.DataFrame(rules)


def rule_synthesiser(rule_df, cutoff=0.5):
  rule_df['class'] = 0
  rule_df['class'].loc[rule_df['score'] >= 0.5] = 1
  
  i = 0
  rm_idx = list()
  while i < len(rule_df) - 1:
    if rule_df['class'][i] == rule_df['class'][i+1]:
      first = rule_df['condition'][i].split(', ')
      second = rule_df['condition'][i+1].split(', ')
      if len(first) == len(second):
        diff = list(filter(lambda x: x[0] != x[1], list(zip(first, second))))
        if len(diff) == 1:
          div = list(map(lambda x: list(map(lambda y: y.strip(' ').strip('= '), x.split(' <'))), diff[0]))
          if len(div[0]) == len(div[1]):
            if len(div[0]) == 2:
              rule_df['condition'][i] = ', '.join(first[1:])
            else:
              if div[0][0] == div[1][2]:
                added = div[1][0] + ' < ' + div[1][1] + ' <= ' + div[0][2]
              elif div[0][2] == div[1][0]:
                added = div[0][0] + ' < ' + div[0][1] + ' <= ' + div[1][2]
              rule_df['condition'][i] = ', '.join([added] + first[1:]) 
          else:
            if len(div[0]) == 3:
              base = div[0]
              com = div[1]
            elif len(div[1]) == 3:
              base = div[1]
              com = div[0]
            in_tf = list(map(lambda x: x in com, base))
            if in_tf[0] == True:
              added = ' <= '.join(base[1:])
            else:
              added = ' < '.join(base[:-1])
            rule_df['condition'][i] = ', '.join([added] + first[1:])
          def rewrite_row(cols, ii):
            for c in cols:
              if c == 'ratio':
                tmp = float(rule_df[c][ii][:-1]) + float(rule_df[c][ii+1][:-1])
                rule_df[c][ii] = "{0:.1f}".format(tmp) + '%'
              elif c == 'score':
                rule_df['score'][ii] = rule_df['sample1'][ii] / (rule_df['sample0'][ii] + rule_df['sample1'][ii])
              elif c == 'impurity':
                rule_df['impurity'][i] = "{0:.3f}".format(rule_df['score'][i] * (1 - rule_df['score'][i]) * 2)
              else:
                rule_df[c][ii] = float(rule_df[c][ii]) + float(rule_df[c][ii+1])
          rewrite_row(['ratio', 'sample0', 'sample1', 'score', 'impurity'], i)
          rm_idx.append(i+1)
          i += 1
    i += 1
  return rule_df.drop(rm_idx).reset_index(drop=True)


def tree_scorer(rule_df, root, cutoff=0.5):
  while True:
    new_df = rule_synthesiser(rule_df, cutoff)
    if len(new_df) == len(rule_df):
      root_rule = ' <= '.join(root)
      var_cut_i = root_rule.index('=')
      i = 0
      left_con = [[1], [1]]
      last_class = -1
      while True:
        if last_class == rule_df['class'][i]:
          left_con[last_class][-1] += 1
        else:
          last_class = rule_df['class'][i]
          left_con[last_class].append(1)
          
        if root_rule in rule_df['condition'][i]:
          i += 1
        elif root_rule[:var_cut_i] in rule_df['condition'][i]:
          ii = rule_df['condition'][i].index(root_rule[:var_cut_i]) + var_cut_i
          try:
            iii = rule_df['condition'][i][ii:].index(', ') + ii
          except ValueError:
            iii = len(rule_df['condition'][i])
          if float(root_rule[var_cut_i+1:]) >= float(rule_df['condition'][i][ii+2:iii]):
            i += 1
          else:
            break
        else:
          break
      mid_last = i - 1
      
      last_class = -1
      right_con = [[1], [1]]
      for i in range(mid_last+1, len(rule_df)):
        if last_class == rule_df['class'][i]:
          right_con[last_class][-1] += 1
        else:
          last_class = rule_df['class'][i]
          right_con[last_class].append(1)
      
      def find_class_maxlen(con):
        maxlen = list(map(lambda x: max(x), con))
        idx = maxlen.index(max(maxlen))
        return (idx, maxlen[idx])
      left_con = find_class_maxlen(left_con)
      right_con = find_class_maxlen(right_con)
      
      var_points = find_split_points(rule_df)
      bunja = np.log(sum(map(lambda x: len(var_points[x]), var_points.keys())))
      score = 10
      if left_con[0] != right_con[0]:
        score += min([(left_con[1] + right_con[1]) * 10, 100])
      else:
        score += abs(left_con[1] - right_con[1]) * 10
      score /= (bunja + len(rule_df))
      
      return rule_df.sort_values(by='class')[['condition', 'score', 'impurity', 'ratio', 'class']], score
    else:
      rule_df = new_df


def find_split_points(rule_df):
  var_points = dict()
  for i in range(len(rule_df)):
    div_condition = rule_df['condition'][i].split(', ')
    for j in range(len(div_condition)):
      div = list(map(lambda x: x.strip(' ').strip('= '), div_condition[j].split('<')))
      try:
        num = float(div[0])
        var_i = 1
      except ValueError:
        var_i = 0
      if div[var_i] not in var_points.keys():
        var_points[div[var_i]] = list()
      for k in range(len(div)):
        if k != var_i:
          var_points[div[var_i]].append(float(div[k]))
  for var in var_points.keys():
    var_points[var] = sorted(set(var_points[var]))
  return var_points
          

def count_by_cutoff(data_df, rule_df):
  var_points = find_split_points(rule_df)
  
  l = len(data_df)
  cnts = list()
  for var in var_points.keys():
    for i in range(len(var_points[var])):
      tmp = sum(data_df[var] <= var_points[var][i])
      cnts.append([var, var_points[var][i], tmp, l - tmp])
  cnts = pd.DataFrame(cnts, columns=['feature', 'cutoff', ' <= ', ' > '])
  
  return cnts


def random_forest_with_performance(train_X_y, n_estimators, max_depth, min_sample_leaf, ratio_features=0.8, valid_X_y=None):
    X, y = train_X_y
    if valid_X_y is None:
      vX = X
      vy = y
    else:
      vX, vy = valid_X_y
    rf = RandomForestClassifier(class_weight='balanced_subsample', max_features=ratio_features, 
                                n_estimators=n_estimators, max_depth=max_depth, 
                                min_samples_leaf=min_sample_leaf, random_state=1234).fit(X, y)
    pred = rf.predict(vX)
    prob = rf.predict_proba(vX)[:, 1]
    
    performance = compute_performance(vy, pred, prob)
    
    features = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values(by='importance', ascending=False)
    features = features.loc[features['importance'] > 0.0]

    result = dict()
    result['model'] = rf
    result['feature importance'] = features
    result['performance'] = performance
    result['columns'] = X.columns

    return result


def select_important_variables(Xy, topN, n_estimators, max_depth, min_sample_leaf, ratio_features):
    rr = random_forest_with_performance(Xy, n_estimators, max_depth, min_sample_leaf, ratio_features)
    return rr['feature importance']['feature'].values[:topN]


def train_and_filter_models(train_Xy, col_list, depth_list, sample_leaf, min_auc, event, duration, max_pvalue, valid_X=None):
    X, y = train_Xy
    if valid_X is None:
        vX = X
    else:
        vX = valid_X
    X_list = list(map(lambda x: X[x], col_list))
    models = list()
    for d in depth_list:
        mds = sequential_tree_modeling(X_list, y, sample_leaf, d, class_weight='balanced')
        mds = list(filter(lambda x: x['performance']['AUC'] >= min_auc, mds))
        preds = list(map(lambda x: x['model'].predict(vX[x['columns']]), mds))
        ps = list(map(lambda x: logrank_pvalue(duration, x, event), preds))
        mds = list(filter(lambda x: x[1] <= max_pvalue, zip(mds, ps)))
        models += list(map(lambda x: x[0], mds))
        print(len(models))
    
    return models


def select_models(models, min_score, print_score=False):
    rule_list = list()
    score_df = list()
    for md in models:
        tree = md['model']
        cols = md['columns']
        rules, root = tree_to_code(tree, cols)
        rules = tree_rule_summariser(rules)
        rules, score = tree_scorer(rules, root)
        rule_list.append(rules)
        score_df.append(score)
    score_df = pd.DataFrame({'model_score': score_df}).sort_values(by='model_score', ascending=False)
    score_df = score_df.loc[score_df['model_score'] >= min_score]
    print(len(score_df))

    if print_score:
        print(score_df)

    selected_models = list()
    for i, j in enumerate(score_df.index):
        models[j]['rules'] = rule_list[j]
        models[j]['model_score'] = score_df.iloc[i]['model_score']
        selected_models.append(models[j])

    return selected_models