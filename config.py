#!/usr/bin/env python
#-*-coding:utf-8-*-
'''
Description:≈‰÷√Œƒº˛
Created on 2016/5/31
@author: zenwan
@version: 
'''
# config the logging system
import logging
logging.basicConfig(filename = "didi_tech.log",\
                    level = logging.DEBUG,\
                    filemode = 'w',
                    format = '%(asctime)s %(name)s [line:%(lineno)d] %(levelname)s %(message)s'
                    )
#input_data_path
train_input_data_path = "E:\\Desktop\\Desktop\\data\\citydata\\season_1\\training_data\\"
test_input_data_path = "E:\\Desktop\\Desktop\\data\\citydata\\season_1\\test_set_1\\"

#output_data_path
output_data_path = "E:\\Desktop\\Desktop\\data\\"

#table_name
table_name = ['order_data','poi_data','traffic_data','weather_data','cluster_map']

#column_name
order_data_names = ['order_id','driver_id','passenger_id','start_district_hash','dest_district_hash','Price','Time']
cluster_map_names = ['district_hash','district_id']
poi_data_names = ['district_hash','poi_class']
traffic_data_names = ['district_hash','tj_level','tj_time']
weather_data_names = ['time','weather','temperature','pm']