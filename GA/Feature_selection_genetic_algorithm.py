# -*- encoding: utf-8 -*-

import random
import math
import numpy as np
# import lightgbm as lgb
from keras.layers import *
import pandas as pd
from Genetic_algorithm import GA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class FeatureSelection(object):
    def __init__(self, aLifeCount=10):
        # self.columns = ['target', 'credit_score', 'overdraft', 'quota', 'quota_is_zero', 'quota_surplus', 'quota_rate', 'credit_score_rank', 'all_is_null_x', 'all_is_zero', 'credit_score_is_null', 'quota_surplus_is_null', 'unit_price_mean', 'unit_price_max', 'unit_price_min', 'unit_price_std', 'record_is_unique', 'auth_id_card_is_null', 'auth_time_is_null', 'phone_is_null', 'all_is_null_y', 'all_not_null', 'card_time_is_null', 'time_phone_is_null', 'feature_1', 'register_days', 'day_mean', 'day_max', 'day_min', 'order_record_count', 'order_record_unique']
        # self.columns = ['target', 'credit_score', 'overdraft', 'quota', 'quota_is_zero', 'quota_surplus', 'quota_rate', 'credit_score_rank', 'all_is_null_x', 'all_is_zero', 'credit_score_is_null', 'quota_surplus_is_null', 'unit_price_mean', 'unit_price_max', 'unit_price_min', 'unit_price_std', 'order_all_is_null', 'amt_order_mean', 'amt_order_max', 'amt_order_min', 'amt_order_std', 'type_pay_count', 'sts_order_count', 'order_phone_count', 'name_rec_md5_count', '货到付款', '在线+京券支付', '上门自提', '余额+限品东券', '在线+余额', '在线+全品东券', 'null_x', '定向京券支付', '高校代理-自己支付', '京豆混合支付', '在线+全品京券', '在线支付', '在线', '京豆支付', '在线+京豆', '前台自付', '白条支付', '在线+东券', '混合支付', '在线+限品东券', '余额', '积分支付', '京券全额支付', '定向京券', '分期付款', '在线支付 ', '邮局汇款', '高校代理-代理支付', '在线预付', '分期付款(招行)', '定向东券', '京豆东券混合支付', '东券混合支付', '在线+东券支付', '京豆', '在线+余额+限品东券', '在线+定向东券', '全品京券', '限品京券', '在线+限品京券', '公司转账', '京券混合支付', 'type_pay_len', '已晒单', '已完成', '等待审核', '已收货', '预订结束', '抢票已取消', 'null_y', '订单已取消', '正在处理', '配送退货', '预约完成', '未抢中', '充值成功', '未入住', '请上门自提', '等待付款', '已退款', '等待付款确认', '已取消', '下单失败', '失败退款', '商品出库', '缴费成功', '已取消订单', '出票成功', '购买成功', '付款成功', '等待收货', '等待处理', '等待退款', '充值失败;退款成功', '退款成功', '完成', '退款完成', '出票失败', '订单取消', '充值失败', '正在出库', 'sts_order_len', 'birthday_is_zero', 'sex_not_male', 'female', 'male', 'sex_secret', 'merriage1', 'merriage2', 'merriage3', 'merriage_is_null', 'account_grade1', 'account_grade2', 'account_grade3', 'account_grade4', 'account_grade5', 'account_grade_is_null', 'qq_bound_is_null', 'wechat_bound_is_null', 'degree', 'id_card_is_null', 'income1', 'income2', 'income3', 'income4', 'income5', 'age_one', 'age_two', 'age_three', 'age_four', 'age_five', 'all_null', 'record_count', '湖南', '河北', '甘肃', '北京', '香港', '陕西', '安徽', '山东', '天津', 'null', '青海', '吉林', '江西', '海南', '重庆', '台湾', '新疆', '辽宁', '广东', '上海', '河南', '广西', '贵州', '山西', '四川', '内蒙', '浙江', '黑龙', '湖北', '宁夏', '福建', '江苏', '云南', '西藏', 'province_len', 'phone_count', 'card_record_count', 'store_card_count', 'have_credit_card', 'card_category_count', 'credit_count', 'card_count_one', 'record_is_unique', 'auth_id_card_is_null', 'auth_time_is_null', 'phone_is_null', 'all_is_null_y', 'all_not_null', 'card_time_is_null', 'time_phone_is_null', 'feature_1', 'register_days', 'day_mean', 'day_max', 'day_min', 'order_record_count', 'order_record_unique']
        self.columns = ['target', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                        'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                        'diluted wines', 'Proline']
        self.data = pd.read_csv(r'dataSet/wine.data', header=None, low_memory=False, names=self.columns)
        print(self.data.values)
        self.train_data, self.validate_data = train_test_split(self.data, shuffle=True, test_size=0.2,
                                                               )
        # self.validate_data = pd.read_csv(r'dataSet/validate_feature.csv', low_memory=False, usecols=self.columns)                  # 由于特征数量较多，这里只读取了上面的部分特征 #
        self.lifeCount = aLifeCount
        self.ga = GA(aCrossRate=0.7,
                     aMutationRage=0.1,
                     aLifeCount=self.lifeCount,
                     aGeneLenght=len(self.columns) - 1,
                     aMatchFun=self.matchFun())

    def all_score(self):
        features = self.columns[1:]
        train_inputs = self.train_data[features].values
        print(np.array(self.train_data['target'], dtype=np.int8) - 1)
        train_labels = to_categorical(np.array(self.train_data['target'], dtype=np.int8) - 1)
        valid_inputs = self.validate_data[features].values
        valid_labels = to_categorical(np.array(self.validate_data['target'], dtype=np.int8) - 1)
        self.callbacks = [
            ModelCheckpoint(
                filepath='weights_all.hdf5',
                monitor='val_acc',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )]
        self.callbacks.append(
            EarlyStopping(
                monitor='val_acc',
                patience=50,
                mode='max'
            ))
        model = self.build_model(len(features), 3)
        model.fit(train_inputs, train_labels, batch_size=20, epochs=1000, verbose=1, callbacks=self.callbacks,
                  validation_data=(valid_inputs, valid_labels))
        model.load_weights('weights_all.hdf5')
        valid_pred = model.predict(valid_inputs)
        # valid_pred = [1 if valid_pred[i] >= 0.5 else 0 for i in range(len(valid_pred))]
        valid_pred = np.argmax(valid_pred, axis=-1)
        acc = accuracy_score(np.array(self.validate_data['target'], dtype=np.int8) - 1, valid_pred)
        print(acc)
        return acc

    # def auc_score(self, order):
    #     print(order)
    #     features = self.columns[1:]
    #     features_name = []
    #     for index in range(len(order)):
    #         if order[index] == 1:
    #             features_name.append(features[index])
    #
    #     labels = np.array(self.train_data['target'], dtype=np.int8)
    #     d_train = lgb.Dataset(self.train_data[features_name], label=labels)
    #     params = {
    #         'boosting': 'gbdt',
    #         'objective': 'binary',
    #         'metric': 'auc',
    #         'train_metric': False,
    #         'subsample': 0.8,
    #         'learning_rate': 0.05,
    #         'num_leaves': 96,
    #         'num_threads': 4,
    #         'max_depth': 5,
    #         'colsample_bytree': 0.8,
    #         'lambda_l2': 0.01,
    #         'verbose': -1,     # inhibit print info #
    #     }
    #     rounds = 100
    #     watchlist = [d_train]
    #     bst = lgb.train(params=params, train_set=d_train, num_boost_round=rounds, valid_sets=watchlist, verbose_eval=10)
    #     predict = bst.predict(self.validate_data[features_name])
    #     print(features_name)
    #     score = roc_auc_score(self.validate_data['target'], predict)
    #     print('validate score:', score)
    #     return score

    def build_model(self, feature_len, feature_len_all):
        features = Input(shape=(feature_len,), dtype='float32')
        x = Dense(50, activation='tanh')(features)
        x = Dense(feature_len_all, activation='softmax')(x)
        model = Model(features, x)
        # model.compile(
        #     loss='mean_squared_error',
        #     optimizer='adam',
        #     metrics=['mae', 'acc']
        # )
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc']
        )
        return model

    def r_m_score(self, order):
        print(order)
        features = self.columns[1:]
        features_name = []
        for index in range(len(order)):
            if order[index] == 1:
                features_name.append(features[index])
        train_inputs = self.train_data[features_name].values
        train_outputs = self.train_data[features].values
        valid_inputs = self.validate_data[features_name].values
        valid_outputs = self.validate_data[features].values
        self.callbacks = [
            ModelCheckpoint(
                filepath='weights.hdf5',
                monitor='val_mean_absolute_error',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )]
        self.callbacks.append(
            EarlyStopping(
                monitor='val_mean_absolute_error',
                patience=6,
                mode='min'
            ))
        model = self.build_model(len(features_name), len(features))
        model.fit(train_inputs, train_outputs, batch_size=5, epochs=100, verbose=1, callbacks=self.callbacks,
                  validation_data=(valid_inputs, valid_outputs))
        valid_pred = model.predict(valid_inputs)
        r = (len(features) - len(features_name) * 2) / len(features)
        r = 1 / (1 + np.exp(-r))
        m = (-np.log(np.sum(np.sum(np.square(valid_pred - valid_outputs), axis=-1), axis=0) / len(valid_pred)) - 6) / 2
        m = 1 / (1 + np.exp(-m))
        return r + m

    def acc_score(self, order):
        print(order)
        features = self.columns[1:]
        features_name = []
        for index in range(len(order)):
            if order[index] == 1:
                features_name.append(features[index])
        train_inputs = self.train_data[features_name].values
        train_labels = to_categorical(np.array(self.train_data['target'], dtype=np.int8)-1)
        valid_inputs = self.validate_data[features_name].values
        valid_labels = to_categorical(np.array(self.validate_data['target'], dtype=np.int8)-1)
        self.callbacks = [
            ModelCheckpoint(
                filepath='weights.hdf5',
                monitor='val_acc',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            )]
        self.callbacks.append(
            EarlyStopping(
                monitor='val_acc',
                patience=50,
                mode='max'
            ))
        model = self.build_model(len(features_name), 3)
        model.fit(train_inputs, train_labels, batch_size=20, epochs=1000, verbose=0, callbacks=self.callbacks,
                  validation_data=(valid_inputs, valid_labels))
        model.load_weights('weights.hdf5')
        valid_pred = model.predict(valid_inputs)
        # valid_pred = [1 if valid_pred[i] >= 0.5 else 0 for i in range(len(valid_pred))]
        valid_pred = np.argmax(valid_pred, axis=-1)
        acc = accuracy_score(np.array(self.validate_data['target'], dtype=np.int8)-1, valid_pred)
        print(acc)
        return acc

    def matchFun(self):
        # return lambda life: self.auc_score(life.gene)
        return lambda life: self.acc_score(life.gene)

    def run(self, n=0):
        distance_list = []
        generate = [index for index in range(1, n + 1)]
        while n > 0:
            self.ga.next()
            # distance = self.auc_score(self.ga.best.gene)
            distance = self.ga.score                      ####
            distance_list.append(distance)
            print(("第%d代 : 当前最好特征组合的线下验证结果为：%f") % (self.ga.generation, distance))
            n -= 1

        print('当前最好特征组合:')
        string = []
        flag = 0
        features = self.columns[1:]
        for index in self.ga.gene:                                  ####
            if index == 1:
                string.append(features[flag])
            flag += 1
        print(string)
        print('线下最高为acc：', self.ga.score)                      ####

        '''画图函数'''
        plt.plot(generate, distance_list)
        plt.xlabel('generation')
        plt.ylabel('distance')
        plt.title('generation--r+m-score')
        plt.show()


def main():
    fs = FeatureSelection(aLifeCount=20)
    rounds = 100    # 算法迭代次数 #
    fs.run(rounds)
    # fs.all_score()


if __name__ == '__main__':
    main()


