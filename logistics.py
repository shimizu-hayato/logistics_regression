#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from scipy import optimize

plt.style.use('ggplot')

def make_data(N, draw_plot=True, is_confused=False, confuse_bin=50):
    '''N個のデータセットを生成する関数
    データをわざと複雑にするための機能 is_confusedを実装する
    '''
    np.random.seed() # シードを固定して、乱数が毎回同じ出力になるようにする

    feature = np.random.randn(N, 2)
    df = pd.DataFrame(feature, columns=['x', 'y'])

    # 2値分類の付与：人為的な分離線の上下どちらに居るかで機械的に判定
    df['c'] = df.apply(lambda row : 1 if (5*row.x + 3*row.y - 1)>0 else 0,  axis=1)

    # 撹乱:データを少し複雑にするための操作
    if is_confused:
        def get_model_confused(data):
            c = 1 if (data.name % confuse_bin) == 0 else data.c 
            return c

        df['c'] = df.apply(get_model_confused, axis=1)

    # 可視化：どんな感じのデータになったか可視化するモジュール
    # c = df.c つまり2値の0と1で色を分けて表示するようにしてある
    if draw_plot:
        plt.scatter(x=df.x, y=df.y, c=df.c, alpha=0.6)
        plt.xlim([df.x.min() -0.1, df.x.max() +0.1])
        plt.ylim([df.y.min() -0.1, df.y.max() +0.1])
        plt.savefig('init.png')
    return df

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z)) if z > -709 else 0.001


def get_prob(x, y, weight_vector):
    '''特徴量と重み係数ベクトルを与えると、確率p(c=1 | x,y)を返す関数
    '''
    feature_vector =  np.array([x, y, 1])
    z = np.inner(feature_vector, weight_vector)

    return sigmoid(z)

def define_likelihood(weight_vector, *args):
    '''dfのデータセット分をなめていき、対数尤度の和を定義する関数
    この関数をOptimizerに喰わせてパラメータの最尤推定を行う    
    '''
    
    likelihood = 0
    df_data = args[0]

    for x, y, c in zip(df_data.x, df_data.y, df_data.c):
        prob = get_prob(x, y, weight_vector)

        i_likelihood = np.log(prob) if c==1 else np.log(1.0-prob)
        likelihood = likelihood - i_likelihood

    return likelihood

def estimate_weight(df_data, initial_param):
    '''学習用のデータとパラメータの初期値を受け取って、
    最尤推定の結果の最適パラメータを返す関数
    '''        
    parameter = optimize.minimize(define_likelihood,
                                  initial_param, #適当に重みの組み合わせの初期値を与える
                                  args=(df_data),
                                  method='Nelder-Mead')

    return parameter.x

def draw_split_line(weight_vector):
    '''分離線を描画する関数
    '''
    a,b,c = weight_vector
    x = np.array(range(-10,10,1))
    y = (a * x + c)/-b
    plt.plot(x,y, alpha=0.3)
    plt.savefig('draw_line.png')
    
def validate_prediction(df_data, weight_vector):

    a, b, c = weight_vector
    df_data['pred'] = df_data.apply(lambda row : 1 if (a*row.x + b*row.y + c) >0 else 0, axis=1)
    df_data['p'] = df_data.apply(lambda row :  sigmoid(a*row.x + b*row.y + c), axis=1)

    return df_data

def draw_prob(df_data):

    df = validate_prediction(df_data, weight_vector)
    plt.scatter(df_data.x, df_data.y, c=df_data.p, cmap='Blues', alpha=0.6)
    plt.xlim([df_data.x.min() -0.1, df.x.max() +0.1])
    plt.ylim([df_data.y.min() -0.1, df.y.max() +0.1])
    plt.colorbar()

    plt.title('plot colored by probability', size=16)
    plt.savefig('graf.png')

if __name__ == '__main__':
    df_data = make_data(1000)
    weight_vector = np.random.rand(3)
    weight_vector = estimate_weight(df_data, weight_vector) #最尤推定の実行
    draw_split_line(weight_vector)

    #最尤推定で重みを推定し、分離面を描画してみる
    weight_vector = estimate_weight(df_data, weight_vector)
    draw_split_line(weight_vector)
    plt.title('plot with split line before/after optimization', size=16)
    
    #pの可視化
    draw_prob(df_data)
