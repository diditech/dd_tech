from __future__ import division
__author__ = 'Hua Wei'
import pandas as pd
import numpy as np
import psycopg2
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
import utils
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
from sklearn import metrics

print("------------------Load Files!")
district_df=utils.generate_distict_df()

training_order_df=utils.generate_order_df(aim="train")

#training_order_df['start_dist_id']=training_order_df['start_dist_hash'].apply(lambda x: int(utils.districthash2int(district_df,x)))
print("------------------Order DataFrame PreProcessing!")
district_df_new=district_df.reset_index()
district_df_new.columns=['start_dist_hash','start_dist_id']
training_order_df=pd.merge(training_order_df,district_df_new,how='left',on=['start_dist_hash'])

district_df_new.columns=['dest_dist_hash','dest_dist_id']
training_order_df=pd.merge(training_order_df,district_df_new,how='left',on=['dest_dist_hash'])

#16:23
#training_order_df['time_of_day']=training_order_df['time'].apply(lambda x: utils.timestring2int(x))
training_order_df['time']=training_order_df['time'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
#16:29
training_order_df['time_of_day']=training_order_df['time'].apply(lambda x:int(x.hour*6+x.minute/10)+1)

training_order_df['day_of_week']=training_order_df['time'].apply(lambda x:datetime.weekday(x))

training_order_df['date']=training_order_df['time'].apply(lambda x: x.strftime("%Y-%m-%d"))

print("------------------Request and Answer DataFrame Generating!")
request=training_order_df.groupby(['date','time_of_day','start_dist_id'])['order_id'].count().reset_index()
request.columns=['date','time_of_day','start_dist_id','request']

answer_training_order_df=training_order_df.dropna(axis=0,subset=['driver_id'])
answer=answer_training_order_df.groupby(['date','time_of_day','start_dist_id'])['order_id'].count().reset_index()
answer.columns=['date','time_of_day','start_dist_id','answer']

request_answer=pd.merge(request,answer,how='left',on=['date','time_of_day','start_dist_id'])
request_answer.fillna(0,inplace=True)
request_answer['gap']=request_answer['request']-request_answer['answer']

request_answer['date']=request_answer['date'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d"))
request_answer['day_of_week']=request_answer['date'].apply(lambda x:datetime.weekday(x))

utils.save_pickle(request_answer,"request_answer")

print("------------------Model Training!")
data_X=request_answer[['time_of_day','start_dist_id','day_of_week']]
data_Y=request_answer['gap']

X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(data_X,data_Y,test_size=0.2,random_state=0)

#training_order_df=utils.generate_order_df(aim="predict")
pred_df=pd.read_csv("./data/test_set_1/read_me_1.txt",sep='\t', names=['origin'])
pred_df['time_of_day']=pred_df['origin'].apply(lambda x: int(x[11:]))
pred_df['date']=pred_df['origin'].apply(lambda x: datetime.strptime(x[:10],"%Y-%m-%d"))
pred_df['day_of_week']=pred_df['date'].apply(lambda x:datetime.weekday(x))

temp=[]
for i in np.arange(start=1,stop=67,step=1):
    pred_df_i=pred_df[:]
    pred_df_i['district_id']=i
    temp.append(pred_df_i)

pred_df=pd.concat(temp,axis=0)

def GBDT():
    loss='ls'
    learning_rate=0.1
    n_estimators=200
    max_depth=5
    subsample=0.5
    gbdt=GradientBoostingRegressor(loss=loss , learning_rate=learning_rate, n_estimators=n_estimators , subsample=1, min_samples_split=2
        , min_samples_leaf=1 , max_depth=max_depth, init=None , random_state=None , max_features=None , alpha=0.9
        , verbose=0, max_leaf_nodes=None, warm_start=False)
    gbdt.fit(X_train,Y_train)
    return gbdt

gbdt=GBDT()

print("------------------Model Testing!")
Y_pred=gbdt.predict(X_test)
Y_pred[Y_pred<0]=0
Y_pred=pd.Series(Y_pred,name='gap_pred')
Y_pred=Y_pred.apply(lambda x:round(x))

X_test_pred= pd.concat([X_test.reset_index(), Y_pred,Y_test.reset_index()['gap']], axis=1)

X_test_pred=X_test_pred[X_test_pred['gap']!=0]
X_test_pred['mape']=abs(X_test_pred['gap']-X_test_pred['gap_pred'])/X_test_pred['gap']
X_test_pred.set_index('index',inplace=True)

result = pd.concat([X_test_pred, request_answer['date']], axis=1, join='inner')

print("MAPE on Testdata:")
print(result.groupby(['start_dist_id'])['mape'].mean().mean())

print("------------------Model Predicting!")
def prediction_gbdt(pred_df,gbdt):
    X_pred=pred_df[['time_of_day','district_id','day_of_week']]
    Y_pred=gbdt.predict(X_pred)
    Y_pred[Y_pred<0]=0
    Y_pred=pd.Series(Y_pred,name='gap_pred')
    Y_pred=Y_pred.apply(lambda x:round(x))
    return Y_pred

Y_pred=prediction_gbdt(pred_df,gbdt)
result_pred= pd.concat([pred_df.reset_index()[['district_id','origin']], Y_pred], axis=1)
utils.result_generate(result_pred,"1st_version")
print("------------------Prediction Done!")
