import pandas as pd
import numpy as np
import random
import scipy.special as special


# 贝叶斯平滑
np.random.seed(0)
class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''moment estimation'''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/(tries[i] + 0.000000001))
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)

        return mean, var/(len(ctr_list)-1)

# 分折计算转化率
# 进行平湖处理
#from smooth import HyperParam
from utility import HyperParam
label_feature = ['title', 'tag', 'prefix', 'max_query_prediction_keys'] + strong_features + add_features
feats = label_feature.copy()
label_feature.append('label')
label_feature.append('n_parts')
data = data_df[label_feature]
df_feature = pd.DataFrame()
data['cnt'] = 1
n_parts = 6
for feat in feats:
    feat_name = feat+'_ctr'
    print(feat_name)
    se = pd.Series()
    for i in range(n_parts):
        if i==0:
            df = data[data['n_parts'] == i+1][[feat]]
            temp = data[(data['n_parts'] != i + 1) & (data['n_parts'] <= 5)][[feat, 'label']].groupby(feat)['label'].agg({feat + '_click': 'sum', feat + '_count': 'count'})
            HP = HyperParam(1, 1)
            HP.update_from_data_by_moment(temp[feat + '_count'].values, temp[feat + '_click'].values)
            temp[feat + '_ctr_smooth'] = (temp[feat + '_click'] + HP.alpha) / (temp[feat + '_count'] + HP.alpha + HP.beta)
            se = se.append(pd.Series(df[feat].map(temp[feat + '_ctr_smooth']).values, index = df.index))
        elif i>=1 and i<=4:
            df = data[data['n_parts']==i+1][[feat]]
            temp = data[(data['n_parts']!=i+1)&(data['n_parts']<=5)&(data['n_parts']>=2)][[feat,'label']].groupby(feat)['label'].agg({feat + '_click': 'sum', feat + '_count': 'count'})
            HP = HyperParam(1, 1)
            HP.update_from_data_by_moment(temp[feat + '_count'].values, temp[feat + '_click'].values)
            temp[feat + '_ctr_smooth'] = (temp[feat + '_click'] + HP.alpha) / (
                        temp[feat + '_count'] + HP.alpha + HP.beta)
            se = se.append(pd.Series(df[feat].map(temp[feat + '_ctr_smooth']).values,index=df.index))
        elif i>=5:
            df = data[data['n_parts']==i+1][[feat]]
            temp = data[data['n_parts']<=5][[feat,'label']].groupby(feat)['label'].agg({feat + '_click': 'sum', feat + '_count': 'count'})
            HP = HyperParam(1, 1)
            HP.update_from_data_by_moment(temp[feat + '_count'].values, temp[feat + '_click'].values)
            temp[feat + '_ctr_smooth'] = (temp[feat + '_click'] + HP.alpha) / (
                    temp[feat + '_count'] + HP.alpha + HP.beta)
            se = se.append(pd.Series(df[feat].map(temp[feat + '_ctr_smooth']).values,index=df.index))
    df_feature[feat_name] = pd.Series(data.index).map(se)
for i in range(len(feats)):
    for j in range(len(feats)-i-1):
        feat_name = feats[i]+"_"+feats[i+j+1]+'_ctr'
        print(feat_name)
        se = pd.Series()
        for k in range(n_parts):
            if k==0:
                temp = data[(data['n_parts']!=k+1)&(data['n_parts']<=5)].groupby([feats[i],feats[i+j+1]])['label'].agg({feat_name + '_click': 'sum', feat_name + '_count': 'count'})
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(temp[feat_name + '_count'].values, temp[feat_name + '_click'].values)
                temp[feat_name + '_ctr_smooth'] = (temp[feat_name + '_click'] + HP.alpha) / (
                        temp[feat_name + '_count'] + HP.alpha + HP.beta)
                dt = data[data['n_parts']==k+1][[feats[i],feats[i+j+1]]]
                dt.insert(0,'index',list(dt.index))
                dt = pd.merge(dt,temp[feat_name + '_ctr_smooth'].reset_index(),how='left',on=[feats[i],feats[i+j+1]])
                se = se.append(pd.Series(dt[feat_name + '_ctr_smooth'].values,index=list(dt['index'].values)))
            elif 1<=k and k<=4:
                temp = data[(data['n_parts']!=k+1)&(data['n_parts']<=5)&(data['n_parts']>=2)].groupby([feats[i],feats[i+j+1]])['label'].agg({feat_name + '_click': 'sum', feat_name + '_count': 'count'})
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(temp[feat_name + '_count'].values, temp[feat_name + '_click'].values)
                temp[feat_name + '_ctr_smooth'] = (temp[feat_name + '_click'] + HP.alpha) / (
                        temp[feat_name + '_count'] + HP.alpha + HP.beta)
                dt = data[data['n_parts']==k+1][[feats[i],feats[i+j+1]]]
                dt.insert(0,'index',list(dt.index))
                dt = pd.merge(dt,temp[feat_name + '_ctr_smooth'].reset_index(),how='left',on=[feats[i],feats[i+j+1]])
                se = se.append(pd.Series(dt[feat_name + '_ctr_smooth'].values,index=list(dt['index'].values)))
            elif k>=5:
                temp = data[data['n_parts']<=5].groupby([feats[i],feats[i+j+1]])['label'].agg({feat_name + '_click': 'sum', feat_name + '_count': 'count'})
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(temp[feat_name + '_count'].values, temp[feat_name + '_click'].values)
                temp[feat_name + '_ctr_smooth'] = (temp[feat_name + '_click'] + HP.alpha) / (
                        temp[feat_name + '_count'] + HP.alpha + HP.beta)
                dt = data[data['n_parts']==k+1][[feats[i],feats[i+j+1]]]
                dt.insert(0,'index',list(dt.index))
                dt = pd.merge(dt,temp[feat_name + '_ctr_smooth'].reset_index(),how='left',on=[feats[i],feats[i+j+1]])
                se = se.append(pd.Series(dt[feat_name + '_ctr_smooth'].values,index=list(dt['index'].values)))
        df_feature[feat_name] = pd.Series(data.index).map(se)
data_df = pd.concat([data_df, df_feature], axis=1)
print('-------------making all sample ctr-----------------')
label_feature=['title', 'tag', 'prefix', 'query_prediction']
col_type = label_feature.copy()
label_feature.append('label')
label_feature.append('n_parts')
data_temp = data_df[label_feature]
df_feature = pd.DataFrame()
data_temp['cnt']=1
se = pd.Series()
for k in range(n_parts):
    if k==0:
        stat = data_temp[(data_temp['n_parts']!=k+1)&(data_temp['n_parts']<=5)].groupby(['prefix', 'query_prediction', 'title', 'tag'])['label'].mean()
        dt = data_temp[data_temp['n_parts']==k+1][['prefix', 'query_prediction', 'title', 'tag']]
        dt.insert(0,'index',list(dt.index))
        dt = pd.merge(dt,stat.reset_index(),how='left',on=['prefix', 'query_prediction', 'title', 'tag'])
        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
    elif 1<=k and k<=4:
        stat = data_temp[(data_temp['n_parts']!=k+1)&(data_temp['n_parts']<=5)&(data_temp['n_parts']>=2)].groupby(['prefix', 'query_prediction', 'title', 'tag'])['label'].mean()
        dt = data_temp[data_temp['n_parts']==k+1][['prefix', 'query_prediction', 'title', 'tag']]
        dt.insert(0,'index',list(dt.index))
        dt = pd.merge(dt,stat.reset_index(),how='left',on=['prefix', 'query_prediction', 'title', 'tag'])
        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
    elif k>=5:
        stat = data_temp[data_temp['n_parts']<=5].groupby(['prefix', 'query_prediction', 'title', 'tag'])['label'].mean()
        dt = data_temp[data_temp['n_parts']==k+1][['prefix', 'query_prediction', 'title', 'tag']]
        dt.insert(0,'index',list(dt.index))
        dt = pd.merge(dt,stat.reset_index(),how='left',on=['prefix', 'query_prediction', 'title', 'tag'])
        se = se.append(pd.Series(dt['label'].values,index=list(dt['index'].values)))
data_df['all_cvr'] = pd.Series(data_temp.index).map(se)
data_df.drop(['n_parts'], axis=1, inplace=True)