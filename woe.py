# author: Jiaxin Peng
# date:   2020.8.20

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
        