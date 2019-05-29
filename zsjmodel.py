import sys
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
import re
import gc
import jieba
import datetime  
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import f1_score
from sklearn import preprocessing
import difflib
import warnings
warnings.filterwarnings("ignore")
start = time.time()

# load data
train_df = pd.read_table('Demo/DataSets/oppo_data_ronud2_20181107/data_train.txt', 
        names= ['prefix','query_prediction','title','tag','label'],quoting=3, header= None, encoding='utf-8').astype(str)
valid_df = pd.read_table('Demo/DataSets/oppo_data_ronud2_20181107/data_vali.txt', 
        names= ['prefix','query_prediction','title','tag','label'],quoting=3, header= None, encoding='utf-8').astype(str)
test_df = pd.read_table('Demo/DataSets/oppo_round2_test_B/oppo_round2_test_B.txt', 
        names= ['prefix','query_prediction','title','tag','label'], quoting=3,header= None, encoding='utf-8').astype(str)

train_data['label'] = train_data['label'].apply(lambda x : int(x))
val_data['label'] = val_data['label'].apply(lambda x : int(x))
test_data['label'] = test_data['label'].apply(lambda x : int(x))

print(train_data.shape, '\n', val_data.shape, '\n', test_data.shape)

train_data['label'] = train_data['label'].apply(lambda x : int(x))
val_data['label'] = val_data['label'].apply(lambda x : int(x))
test_data['label'] = test_data['label'].apply(lambda x : int(x))

train_data['flag'] = 0
val_data['flag'] = 1
test_data['flag'] = -1
data = pd.concat([train_data, val_data, test_data])
# merge_data.label.value_counts()



# dict keys sort
def dict_sort(text):
    try:
        dicts = json.loads(text)
    except:
        dicts = {}
    return sorted(dicts.items(),key = lambda x:float(x[1]),reverse = True)
data['pred_list'] = data['query_prediction'].apply(dict_sort)
data['pred_len'] = data['pred_list'].apply(len)
data['prefix_len'] = data.prefix.apply(len)
data['title_len'] = data.title.apply(len)
data['is_prefix_in_title'] = data[['prefix','title']].apply(lambda row: row[1].find(row[0]),raw=True,axis=1)
data['title-prefix_len'] = data.title_len - data.prefix_len
data['ratio_len'] = data['prefix_len']/data['title_len']

def remove_cha(x):
    x = re.sub('[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', "", str(x))
    x = x.replace('2C', '')
    return x
def get_query_prediction_keys(x):
    try:
        x = json.loads(x)
    except:
        x = {}
    x = x.keys()
    x = [remove_cha(value) for value in x]    
    return ' '.join(x)
data['query_prediction_keys'] = data.query_prediction.apply(lambda x:get_query_prediction_keys(x))

def len_title_in_query(title, query):
    query = query.split(' ')
    if len(query) == 0:
        return 0
    l = 0
    for value in query:
        if value.find(title) >= 0:
            l += 1
    return l
data['is_title_in_query_keys'] = data.apply(lambda row:len_title_in_query(row['title'], row['query_prediction_keys']),axis = 1)
data['is_prefix_in_query_keys'] = data.apply(lambda row:len_title_in_query(row['prefix'], row['query_prediction_keys']),axis = 1)
data = data.drop(['query_prediction_keys'], axis=1)

# 参考https://github.com/luoling1993/TianChi_OGeek/blob/master/stat_engineering.py
def get_max_query_ratio(item):
    query_prediction = item['query_prediction']
    try:
        query_prediction = json.loads(query_prediction)
    except:
        query_prediction = {}
    title = item['title']
    if not query_prediction:
        return 0
    for query_wrod, ratio in query_prediction.items():
        if title == query_wrod:
            if float(ratio) > 0.1:
                return 1
    return 0

def get_word_length(item):
    item = str(item)
    word_cut = jieba.lcut(item)
    length = len(word_cut)
    return length

def get_small_query_num(item):
    small_query_num = 0
    try:
        item = json.loads(item)
    except:
        item = {}    
    for _, ratio in item.items():
        if float(ratio) <= 0.08:
            small_query_num += 1

    return small_query_num

data['max_query_ratio'] = data.apply(get_max_query_ratio, axis=1)
data['prefix_word_num'] = data['prefix'].apply(get_word_length)
data['title_word_num'] = data['title'].apply(get_word_length)
data['small_query_num'] =data['query_prediction'].apply(get_small_query_num)

# title 在 query_prediction 中构造
def is_in_query_loc(lst):
    try:
        pred = eval(lst[1])
    except:
        pred = {}
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    for i,value in enumerate(pred):
        if(lst[0]==value[0]):
            return i
    return -1
data['title_in_query_loc'] = data[['title', 'query_prediction']].apply(is_in_query_loc, axis=1)

def is_in_query(lst):
    try:
        dicts = json.loads(lst[1])
    except:
        dicts = {}
    if lst[0] in dicts.keys():
        return 1
    else:
        return 0
data['title_in_query'] = data[['title', 'query_prediction']].apply(is_in_query, axis=1)

def _in_query_proba(lst):
    try:
        dicts = json.loads(lst[1])
    except:
        dicts = {}
    if lst[0] in dicts.keys():
        return dicts[lst[0]]
    else:
        return -1
data['title_in_query_proba'] = data[['title','query_prediction']].apply(_in_query_proba, axis=1)


# 参考https://github.com/GrinAndBear/OGeek/blob/master/create_feature.py
# 构造相似度列表
def extract_proba(pred):
    try:
        pred = eval(pred)
    except:
        pred = {}
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    pred_proba_lst=[]
    for i in range(10):
        if len(pred)<i+2:
            pred_proba_lst.append(0)
        else:
            pred_proba_lst.append(float(pred[i][1]))
    return pred_proba_lst

def extract_prefix_pred_similarity(lst):
    try:
        pred = eval(lst[1])
    except:
        pred = {}
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    prefix_pred_sim=[]
    for i in range(10):
        if len(pred)<i+2:
            prefix_pred_sim.append(0)
        else:
            prefix_pred_sim.append(difflib_similarity(lst[0],pred[i][0]))
    return prefix_pred_sim

def difflib_similarity(str1,str2):
    return difflib.SequenceMatcher(a=str1, b=str2).quick_ratio()

def extract_title_pred_similarity(lst):
    try:
        pred = eval(lst[1])
    except:
        pred = {}
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    title_pred_sim=[]
    for i in range(10):
        if len(pred)<i+2:
            title_pred_sim.append(0)
        else:
            title_pred_sim.append(difflib_similarity(lst[0],pred[i][0]))
    return title_pred_sim

print('pred proba starting')
data['pred_proba_lst']=data['query_prediction'].apply(extract_proba)
print('prefix pred starting')
data['prefix_pred_sim']=data[['prefix','query_prediction']].apply(extract_prefix_pred_similarity,axis=1)
print('title pred starting')
data['title_pred_sim']=data[['title','query_prediction']].apply(extract_title_pred_similarity,axis=1)


# 对相似度列表进行统计
data['pred_proba_lst_max'] = data['pred_proba_lst'].apply(lambda x: max(x))
data['prefix_pred_sim_max'] = data['prefix_pred_sim'].apply(lambda x: max(x)) 
data['title_pred_sim_max'] = data['title_pred_sim'].apply(lambda x: max(x))
data['pred_proba_lst_std'] = data['pred_proba_lst'].apply(lambda x: np.std(x))
data['prefix_pred_sim_std'] = data['prefix_pred_sim'].apply(lambda x: np.std(x)) 
data['title_pred_sim_std'] = data['title_pred_sim'].apply(lambda x: np.std(x))
data['proba_max_prefix'] = data['prefix_pred_sim'].apply(lambda x: x[0])
data['proba_max_title'] = data['title_pred_sim'].apply(lambda x: x[0])

def do_mean(li):
    sums = 0
    for i in li:
        sums = sums + i
    return sums/len(li)
data['pred_proba_lst_mean'] = data['pred_proba_lst'].apply(do_mean)
data['prefix_pred_sim_mean'] = data['prefix_pred_sim'].apply(do_mean)
data['title_pred_sim_mean'] = data['title_pred_sim'].apply(do_mean)
data['prefix_title_sim']=data[['prefix','title']].apply(lambda row: difflib_similarity(row[0],row[1]),raw=True,axis=1)

# prefix和title的共现词
def word_match_share(row, item1, item2):
    item1_words = {}
    item2_words = {}
    for word in row[item1].split():
        item1_words[word] = 1
    for word in row[item2].split():
        item2_words[word] = 1
    if len(item1_words) == 0 or len(item2_words) == 0:
        return 0
    shared_words_in_item1 = [w for w in item1_words.keys() if w in item2_words]
    shared_words_in_item2 = [w for w in item2_words.keys() if w in item1_words]
    R = (len(shared_words_in_item1) + len(shared_words_in_item2))*1.0/(len(item1_words)+len(item2_words))
    return R
data['title_prefix_common_words'] = data.apply(lambda x: len(set(x['title'].split()).intersection(set(x['prefix'].split()))), axis = 1) # 有区分度

# 计算查询词prefix出现在title中的那个位置，前、后、中、没出现
def get_prefix_loc_in_title(prefix,title):
    if prefix not in title:
        return -1
    lens = len(prefix)
    if prefix == title[:lens]:
        return 0
    elif prefix == title[-lens:]:
        return 1
    else:
        return 2
data['prefix_loc'] = data.apply(lambda x : get_prefix_loc_in_title(x['prefix'],x['title']), axis=1) # 高区分度

data.drop(['label', 'pred_list', 'pred_proba_lst', 'prefix_pred_sim', 'title_pred_sim'], axis=1, inplace=True)
train_data = data[data.flag != -1]
test = data[data.flag == -1]

since = time.time()
# 算一些全局统计量
# ---- click 特征 ----
list_click_feature = ['prefix', 'title', 'tag', 'max_query_prediction_keys']

# 计算某特征单次点击
for feature in list_click_feature:
    printlog('计算' + feature + '点击次数', is_print_output)
    not_zip_all_data[feature + '_click'] = not_zip_all_data.groupby(feature)[feature].transform('count')
# 部分二元交叉点击
not_zip_all_data['prefix_title_click'] = not_zip_all_data.groupby(['prefix', 'title']).prefix.transform('count')
not_zip_all_data['prefix_tag_click'] = not_zip_all_data.groupby(['prefix', 'tag']).prefix.transform('count')
not_zip_all_data['title_tag_click'] = not_zip_all_data.groupby(['title', 'tag']).title.transform('count')
not_zip_all_data['title_max_query_prediction_keys_click'] = not_zip_all_data.groupby(['title', 'max_query_prediction_keys']).title.transform('count')
not_zip_all_data['tag_max_query_prediction_keys_click'] = not_zip_all_data.groupby(['tag', 'max_query_prediction_keys']).tag.transform('count')
# 部分三元交叉点击
not_zip_all_data['prefix_title_tag_click'] = not_zip_all_data.groupby(['prefix', 'title', 'tag']).prefix.transform('count')
items = ['prefix', 'title', 'tag'] # 两两组合计算搜索 点击 转化率
# 各字段统计数目，点击次数，转化率，排名，占比
items = ['prefix', 'title', 'tag']
for item in items:
    # 分别按照prefix title tag进行分组，并对label进行计算 求和 计数 计算转化率
    temp = train_data.groupby(item, as_index = False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
    temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'] + 3) # 平滑
    item = item[[item+'_ctr']]
    train_data = pd.merge(train_data, temp, on=item, how='left')
    test_data_ = pd.merge(test_data, temp, on=item, how='left')

for i in range(len(items)):
    for j in range(i+1, len(items)):
        item_g = [items[i], items[j]]
        temp = train_data_.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'_count':'count'})
        temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'_count'] + 3) # 平滑
        temp = temp[['_'.join(item_g)+'_ctr']]
        train_data_ = pd.merge(train_data_, temp, on=item_g, how='left')
        test_data_ = pd.merge(test_data_, temp, on=item_g, how='left')


# 这里使用了train同时预测 test 和 val 取概率作为新的特征
train = train_data[train_data['flag'] == 0]
val = train_data[train_data['flag'] == 1]

train = train.drop(['flag'], axis=1)
val = val.drop(['flag'], axis=1)
test = test.drop(['flag'], axis=1)

y_train = train['label'].values
X_train = train.drop(['label'], axis=1).values
y_val = val.label.values
X_val = val.drop(['label'], axis=1).values

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 144,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_seed':0,
    'bagging_freq': 1,
    'verbose': 1,
    'reg_alpha':3,
    'reg_lambda':5
}
print('=======================  train+valid训练  =========iter987==========')
lgb_train = lgb.Dataset(X_train, y_train)
gbm = lgb.train(params, lgb_train, valid_sets=[lgb_train], num_boost_round = 987, verbose_eval=50)

# 预测结果并保存
test_df['pred'] = gbm.predict(X_test, num_iteration=987)
val_df['pred'] = gbm.predict(X_test, num_iteration=987)


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 144,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_seed':0,
    'bagging_freq': 1,
    'verbose': 1,
    'reg_alpha':3,
    'reg_lambda':5
}
print('=======================  train+valid训练  =========iter987==========')
lgb_train = lgb.Dataset(X_val, y_val)
gbm = lgb.train(params, lgb_train, valid_sets=[lgb_train], num_boost_round = 987, verbose_eval=50)

test['label'] = gbm.predict(X_test, num_iteration=987)

