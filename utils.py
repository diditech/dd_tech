#!/usr/bin/env python
#-*-coding:utf-8-*-
'''
Description:
Created on 2016Äê5ÔÂ26ÈÕ
@author: weihua
@version: 
'''
from __future__ import division
import pandas as pd
import numpy as np
import pandas.io.sql as sql
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import metrics
from datetime import datetime
from pandas.tseries.offsets import Day
import time
from sklearn.externals import joblib
import cPickle as pkl
import glob
import config

def ts2string(x):
    return x.strftime("%Y-%m-%d %H:%M:%S")


def string2ts(x):
    return datetime.strptime(x,"%Y-%m-%d %H:%M:%S")


def timestring2int(tString):
    ts=string2ts(tString)
    return int(ts.hour*6+ts.minute/10)+1


def cvt_dayofweek(x):
    return datetime.weekday(x)


def generate_distict_df():
    data=pd.read_csv("./data/training_data/cluster_map/cluster_map",sep='\t',names=['district_hash','district_id'],index_col=0)
    output = open("./data/preprocessed_data/district_dict.pkl", 'wb')
    pkl.dump(data,output,True)
    output.close()
    return data


def load_disctrict_df(file="district_dict.pkl"):
    pkl_file = open("./data/preprocessed_data/%s"%file, 'rb')
    data=pkl.load(pkl_file)
    pkl_file.close()
    return data


def districthash2int(data,districtString):
    return data.ix[districtString]['district_id']


def generate_data(column_names = config.order_data_names,aim = "train",table_name = "order_data"):
    if aim == "train":
        input_path = config.train_input_data_path
    elif aim == "test":
        input_path = config.test_input_data_path
    table_path = input_path + table_name
    dirs = glob.glob(table_path+"\\*")
    temp_dfs = []
    for dir in dirs:
        _data = pd.read_csv(dir,sep = "\t",names = column_names)
        temp_dfs.append(_data)
    data = pd.concat(temp_dfs,axis = 0)
    return data


def generate_order_df(aim="train"):
    output = open("./data/preprocessed_data/order_df_%s.pkl"%aim, 'wb')
    if(aim=="train"):
        input_path="training_data"
        ts_start=string2ts("2016-01-01 00:00:00")
        count=0
        temp=[]
        while(count<20):
            ts_start_day=ts2string(ts_start)[:10]
            print("./data/%s/order_data/order_data_%s"%(input_path,ts_start_day))
            data=pd.read_csv("./data/%s/order_data/order_data_%s"%(input_path,ts_start_day),sep='\t',
                             names=['order_id','driver_id','passenger_id','start_dist_hash',
                                    'dest_dist_hash','price','time'])
            temp.append(data)
            ts_start= ts_start + Day()
            count+=1
        data=pd.concat(temp,axis=0)
        pkl.dump(data,output,True)
    elif(aim=="test"):
        input_path="test_set_1"
        test_set=['2016-01-22','2016-01-24','2016-04-26','2016-01-28','2016-01-30']
        temp=[]
        for ts_start_day in test_set:
            data=pd.read_csv("./data/%s/order_data/order_data_%s"%(input_path,ts_start_day),sep='\t',
                             names=['order_id','driver_id','passenger_id','start_dist_hash',
                                    'dest_dist_hash','price','time'])
            temp.append(data)
        data=pd.concat(temp,axis=0)
        pkl.dump(data,output,True)
    elif(aim=="predict"):
        input_path="test_set_1"
        data=pd.read_csv("./data/%s/order_data/read_me_1.txt"%(input_path),sep='\t',
                             names=['origin'])
    output.close()
    return data


def result_generate(result,comment="temp"):
    if(comment=="temp"):
        comment=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    else:
        comment+=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    result.to_csv("./data/result/result%s.csv"%comment,sep=",",header=False)

def save_pickle(obj,file_name="non-named"):
    output = open("./data/preprocessed_data/%s.pkl"%file_name, 'wb')
    pkl.dump(obj,output,True)
    output.close()

if __name__=="__main__":
    pass