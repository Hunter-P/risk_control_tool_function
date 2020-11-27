# author: Jiaxin Peng
# date:   2020.8.19

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

matplotlib.rcParams['font.family']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False


def plot_roc_curve(y_t, y_t_pred, y_v, y_v_pred, y_o, y_o_pred, save_img_path=None):
    """
    绘制roc曲线
    :param y_t: train data 的标签
    :param y_t_pred: train data 的预测概率
    :param y_v: valid data 的标签
    :param y_v_pred: valid data 的预测概率
    :param y_o: oot data 的标签
    :param y_o_pred: oot data 的预测概率
    :param save_img_path: 图片保存路径
    :return: None
    """
    plt.clf()
    fpr, tpr, _ = metrics.roc_curve(y_t, y_t_pred, pos_label=1)
    plt.plot(fpr, tpr, label='train', c='blue')
    fpr, tpr, _ = metrics.roc_curve(y_v, y_v_pred, pos_label=1)
    plt.plot(fpr, tpr, label='valid', c='r')
    fpr, tpr, _ = metrics.roc_curve(y_o, y_o_pred, pos_label=1)
    plt.plot(fpr, tpr, label='OOT', c='g')
    plt.plot([0,1], [0,1], 'k--')

    plt.xlabel('False positive rate', fontsize=13)
    plt.ylabel('True positive rate', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title('ROC Curve', fontsize=13)
    plt.legend(loc='best', fontsize=13)
    if save_img_path is not None:
        plt.savefig(save_img_path, bbos_inches='tight')
    plt.show()


def get_score_1(prob):
    """
    将概率转化为评分
    :param prob: 概率
    :param od:
    :param bins:
    :param base:
    :return:
    """
    prob=np.clip(np.array(prob), 0.00000001, 0.9999999)
    return prob

def get_score_2(prob, od=15, bins=20, base=500):
    """
    将概率转化为评分
    :param prob: 概率
    :param od:
    :param bins:
    :param base:
    :return:
    """
    prob=np.clip(np.array(prob), 0.00000001, 0.9999999)
    odds=(1-prob)/prob
    score=base + bins*(np.log(odds) - np.log(od))/np.log(2)
    return score.tolist()

def plot_ks_curve_1(y_label, y_pred, bin=200, save_img_path=None):
    pred_list=get_score_1(list(y_pred))
    label_list=list(y_label)
    total_pos=sum(label_list)
    total_neg=len(label_list)-total_pos
    
    items=sorted(zip(pred_list, label_list), key=lambda x: x[0])
    step=(max(pred_list) - min(pred_list)) / bin
    
    pred_bin, pos_rate, neg_rate, ks_list=[],[],[],[]
    for i in range(1, bin+1):
        idx=min(pred_list) + i*step
        pred_bin.append(idx)
        label_bin=[x[1] for x in items if x[0]<idx]
        pos_num=sum(label_bin)
        neg_num=len(label_bin)-pos_num
        posrate=pos_num / totoal_pos
        negrate=neg_num / totoal_neg
        ks=abs(posrate-negrate)
        pos_rate.append(posrate)
        neg_rate.append(negrate)
        ks_list.append(ks)
    fig=plt.figure(figsize=(8,5))
    ax=fig.add_subplots(1,1,1)
    ax.plot(pred_bin, pos_rate, 'g--', linewidth=2, label='Posi Cumu')
    ax.plot(pred_bin, neg_rate, 'r--', linewidth=2, label='Neg Cumu')
    
    KS=max(ks_list)
    ks_loc=ks_list.index(max(ks_list))
    height=max(ks_list)
    left=pred_bin[ks_loc]
    bar_width=0.01
    bottom=pos_rate[ks_loc]
    plt.title('K-S Curve (KS={:.1f}, probability={:.2f}'.format(KS*100, left), fontsize=13)
    plt.bar(left, height, bar_width, bottom, alpha=0.5, color='black', label='K-S Value')
    
    plt.grid(alpha=.7)
    plt.legend(loc='best', fancybox=True, shadow=True, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Probability', fontsize=13)
    plt.ylabel('KS Value', fontsize=13)
    plt.xlim([-0.05, 1.1])
    plt.ylim([0.0, 1.1])
    if save_img_path is not None:
        plt.savefig(save_img_path, bbos_inches='tight')
    plt.show()


def plot_ks_curve_2(y_test, y_prob, pos_label=1, figsize=(8,5), save_img_path=None):
    """
    绘制ks曲线
    :param y_test: 样本标签
    :param y_prob: 样本预测的概率
    :param pos_label: 坏样本标签的值
    :param figsize: 图片尺寸
    :param save_img_path: 图片保存路径
    :return: None
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob, pos_label)
    thresholds = get_score_2(thresholds)
    KS = max(list(tpr-fpr))
    ks_loc = list((tpr-fpr).index(KS))
    bar_width = 10
    left = thresholds[ks_loc]
    bottom = list(fpr)[ks_loc]
    height = KS

    plt.clf()
    plt.figure(figsize=figsize)
    plt.plot(thresholds, tpr, 'r--', linewidth=2, label='Posi Cumu Response')
    plt.plot(thresholds, fpr, 'r--', linewidth=2, label='Nega Cumu Response')
    plt.bar(left, height, bar_width, bottom, alpha=0.5, color='black', label='K-S Value', fontsize=13)
    plt.xlabel('Risk Score', fontsize=13)
    plt.ylabel('KS Value', fontsize=13)
    plt.xlim([100, 1000])
    plt.ylim([0.0, 1.02])
    plt.title('K-S Curve (KS={:.0f}'.format(KS*100, thresholds[ks_loc]), fontsize=13)
    plt.grid(alpha=.7)
    plt.legend(loc='best', fancybox=True, shadow=True, fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if save_img_path is not None:
        plt.savefig(save_img_path, bbos_inches='tight')
    plt.show()


def plt_score_dist(y_test, y_pred, pos_label=1, figsize=(8,5), bins_cnt=50, save_img_path=None):
    """
    绘制评分分布图
    :param y_test: 样本标签
    :param y_pred:  样本预测概率
    :param pos_label: 坏样本标签的值
    :param figsize: 图片尺寸
    :param bins_cnt: 分箱箱数
    :param save_img_path: 图片保存路径
    :return: None
    """
    risk_score = get_score(list(y_pred))
    score_label = pd.DataFrame(y_test)
    score_label.columns = ['label']
    score_label['score'] = risk_score
    score_posi = score_label[score_label['label'] == pos_label]
    score_nega = score_label[score_label['label'] != pos_label]

    plt.clf()
    plt.figure(figsize=figsize)
    start = max(0, min(risk_score)-100)
    end = min(1000, max(risk_score) + 100)
    bins = np.linspace(start, end, num=bins_cnt)
    plt.hist(score_posi['score'], bins, rwidth=0.7, color='r', alpha=.7, label='Class Posi',
             weights=np.ones_like(np.array(score_posi['score']))/float(len(np.array(score_posi['score']))))
    plt.hist(score_nega['score'], bins, rwidth=0.7, color='g', alpha=.7, label='Class Nega',
             weights=np.ones_like(np.array(score_nega['score'])) / float(len(np.array(score_nega['score']))))
    plt.grid(alpha=.7)
    plt.legend(loc='best', fancybox=True, shadow=True, fontsize=13)
    plt.xlim([200, 900])
    plt.ylim([0.0, 0.2])
    plt.xlabel('Proportion', fontsize=13)
    plt.ylabel('Risk Score', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('RiskScore Distribution', fontsize=13)
    if save_img_path is not None:
        plt.savefig(save_img_path, bbos_inches='tight')
    plt.show()
