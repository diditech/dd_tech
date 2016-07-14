#!/usr/bin/env python
#-*-coding:utf-8-*-
'''
Description:
Created on 2016-5-26
@author: zenwan
@version: 
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from config import logging
import utils
import config

class LoadData:
    def __init__(self,aim = 'train'):
        self.aim = aim
    def load_data(self,):
logging.info('��ʼ���ض�������!')
df_order_data = utils.generate_data(column_names=config.order_data_names,\
                    aim='train',\
                    table_name='order_data' )
logging.info('�������ݼ������!')
logging.info('��ʼ����poi����!')
df_order_data = utils.generate_data(column_names=config.poi_data_names,\
                    aim='train',\
                    table_name='poi_data' )
logging.info('poi���ݼ������!')
logging.info('��ʼ���ص�·ӵ������!')
df_traffic_data = utils.generate_data(column_names=config.traffic_data_names,\
                    aim='train',\
                    table_name='traffic_data')
logging.info('��·ӵ�����ݼ������!')
logging.info('��ʼ������������!')
df_weather_data = utils.generate_data(column_names=config.weather_data_names,\
                    aim='train',\
                    table_name='weather_data' )
logging.info('�������ݼ������!')
logging.info('��ʼ��������������!')
df_cluster_map = utils.generate_data(column_names=config.cluster_map_names,\
                    aim='train',\
                    table_name='cluster_map' )
logging.info('���������ݼ�����ϣ�')



