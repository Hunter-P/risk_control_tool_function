# author: Jiaxin Peng
# date:   2020.8.20

import math


'''
woe编码
如果是连续型变量，需要先分箱再编码
'''


class charWoe(object):
    def __init__(self, data, label_name, weight, fea):
        """
        :param data: pd.DataFrame
        :param label_name: 标签列名
        :param weight: 权重，可不设
        :param fea: 需要编码的变量列名
        """
        self.data = data
        self.label_name = label_name
        self.vars = fea
        self.nrows, self.cols = self.data.shape
        self.weight = weight

    def char_woe(self):
        dic = dict(self.data.groupby([self.label_name]).size())
        good = dic.get(0, 0) + 1e-10
        bad = dic.get(1, 0) + 1e-10
        for col in self.vars:
            d = dict(self.data[[col, self.label_name]].groupby([col, self.label_name]).size())
            '''
            当特征取值超过100，跳过
            '''
            if len(d)>100:
                print(col, "contains too many different values...")
                continue
            dic = dict()
            for k,v in d.items():
                value, dp = k
                dic.setdefault(value, {})
                dic[value][int(dp)] = v
            for k,v in dic.items():
                dic[k] = {str(int(k1)): v1 for k1,v1 in v.items()}
                dic[k]['cnt'] = sum(v.values())
                bad_rate = round(v.get(1, 0) / dic[k]['cnt'], 5)
                dic[k]['bad_rate'] = bad_rate

            dic = self.combine_box_char(dic)
            '''
            对每个特征计算woe和iv
            '''
            for k,v in dic.items():
                a = v.get('0', 1)/good + 1e-10
                b = v.get('1', 1) / bad + 1e-10
                dic[k]['Good'] = v.get('0', 0)
                dic[k]['Bad'] = v.get('1', 0)
                dic[k]['woe'] = round(math.log(b/a), 5)
                dic[k]['iv'] = (b-a)*dic[k]['woe']

            dic['IV'] = sum([v['iv'] for v in dic.values()])
            return dic

    def combine_box_char(self, dic):
        """
        实施两种分箱策略
        1. 不同箱之间负样本占比差异最大化
        2. 每一箱的样本量不能过少
        :param dic:
        :return:
        """
        '''
        合并至10箱以内，按照每箱负样本差异最大化原则分箱
        '''
        while len(dic)>=10:
            bad_rate_dic = {k:v['bad_rate'] for k,v in dic.items()}
            bad_rate_sorted = sorted(bad_rate_dic.items(), key=lambda x:x[1])
            bad_rate = [bad_rate_sorted[i+1][1] - bad_rate_sorted[i][1] for i in range(len(bad_rate_sorted)-1)]
            min_rate_index = bad_rate.index(min(bad_rate))
            # k1, k2 为差值最小的两箱的key
            k1, k2 = bad_rate_sorted[min_rate_index][0], bad_rate_sorted[min_rate_index+1][0]
            dic['%s,%s' % (k1, k2)] = dict()
            dic['%s,%s' % (k1, k2)]['0'] = dic[k1].get('0', 0) + dic[k2].get('0', 0)
            dic['%s,%s' % (k1, k2)]['1'] = dic[k1].get('1', 0) + dic[k2].get('1', 0)
            dic['%s,%s' % (k1, k2)]['cnt'] = dic[k1]['cnt'] + dic[k2]['cnt']
            dic['%s,%s' % (k1, k2)]['bad_rate'] = round(dic['%s,%s' % (k1, k2)]['1'] / dic['%s,%s' % (k1, k2)]['cnt'], 5)
            del dic[k1], dic[k2]
        '''
        实施第二种分箱策略， 每一箱样本数量太少
        当样本数量小于总样本的5%且箱的个数大于5的时候，对箱进行合并
        '''
        min_cnt = min(v['cnt'] for v in dic.values())  # 当前最小箱的样本个数
        while min_cnt < self.nrows*0.05 and len(dic)>5:
            min_key = [k for k,v in dic.items() if v['cnt']==min_cnt][0]
            bad_rate_dic = {k: v['bad_rate'] for k,v in dic.items()}
            bad_rate_sorted = sorted(bad_rate_dic.items(), key=lambda x:x[1])
            keys = [k[0] for k in bad_rate_sorted]
            min_index = keys.index(min_key)
            # 保持合并后箱之间的负样本占比差异最大化
            if min_index==0:
                k1, k2 = keys[:2]
            elif min_index==len(dic)-1:
                k1, k2 = keys[-2:]
            else:
                bef_bad_rate = dic[min_key]['bad_rate'] - dic[keys[min_index-1]]['bad_rate']
                aft_bad_rate = dic[keys[min_index+1]]['bad_rate'] - dic[min_key]['bad_rate']
                if bef_bad_rate < aft_bad_rate:
                    k1,k2 = keys[min_index-1], min_key
                else:
                    k1,k2 = min_key, keys[min_index+1]
            dic['%s,%s' % (k1, k2)] = dict()
            dic['%s,%s' % (k1, k2)]['0'] = dic[k1].get('0', 0) + dic[k2].get('0', 0)
            dic['%s,%s' % (k1, k2)]['1'] = dic[k1].get('1', 0) + dic[k2].get('1', 0)
            dic['%s,%s' % (k1, k2)]['cnt'] = dic[k1]['cnt'] + dic[k2]['cnt']
            dic['%s,%s' % (k1, k2)]['bad_rate'] = round(dic['%s,%s' % (k1, k2)]['1'] / dic['%s,%s' % (k1, k2)]['cnt'], 5)
            del dic[k1], [k2]
            min_cnt = min([v['cnt'] for v in dic.values()])

        return dic





