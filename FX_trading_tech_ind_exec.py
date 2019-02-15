# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 18:50:32 2019

@author: jason
"""

import os
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statistics as stat
from scipy import stats
from scipy import linalg
import itertools
import sys
import json
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20 import API
#len(all_ccy_ML)

master_path= r'C:\Users\jason\OneDrive\Documents\Quantopian\FX Trading\ML Result Table'
all_ccy_ML= os.listdir(master_path)
access_token = "10277bb95d0ce16ea52363a0e4e0a54b-b1ad5848ee91d9809c43b7dcee75e537"

ccy=[]

pca_final=[]
num_kNN_final=[]
accur_final=[]

for num_seq in range(len(all_ccy_ML)):
    #i=0
    pca=[]
    num_kNN=[]
    accuracy=[]

    model= pd.read_csv(master_path + '\\' + all_ccy_ML[num_seq])
    n_kNN = model['n_kNN']
    even=[]
    
    for i in range(len(n_kNN)):
        if (n_kNN[i] % 2 ==0):
            even.append(1)
        else:
            even.append(0)
    
    model['even_neighbour']= even
    
    model_filtered=[]
    
    for x in range(len(model['even_neighbour'])):
        if model['even_neighbour'][x]==0:
           pca.append(model['n_pca'][x])
           num_kNN.append(model['n_kNN'][x])
           accuracy.append(model['accuracy'][x])
         
    ccy.append(all_ccy_ML[num_seq][0:6])
    pca_final.append(pca[0])
    num_kNN_final.append(num_kNN[0])
    accur_final.append(accuracy[0])

model_final= pd.concat([pd.DataFrame(ccy),pd.DataFrame(pca_final),#
                        pd.DataFrame(num_kNN_final),pd.DataFrame(accur_final)],axis=1)

model_final.columns=['CCY','N_PCA','N_kNN','Accuracy']

################################################################################################
master_path= r'C:\Users\jason\OneDrive\Documents\Quantopian\FX Trading\G10 Historical Rates\Processed Data'
all_ccy= os.listdir(master_path)

def long_short_SMA(dat,n,N):
    ccy_price= dat
    sma_short=[]
    sma_long=[] 
    
    for i in range(n,len(ccy_price)):
        sma_short.append(np.mean(ccy_price[i- n:i]))
    
    for i in range(N,len(ccy_price)):
        sma_long.append(np.mean(ccy_price[i- N:i]))
    
    sma_short_try= pd.concat([pd.DataFrame(np.repeat(None,n)),pd.DataFrame(sma_short)],axis=0)
    sma_short_try.index= range(len(ccy_price))
    sma_short_try.columns= ['SMA_Short']
    
    sma_long_try= pd.concat([pd.DataFrame(np.repeat(None,N)),pd.DataFrame(sma_long)],axis=0)
    sma_long_try.index= range(len(ccy_price))
    sma_long_try.columns= ['SMA_Long']
    
    return sma_short_try, sma_long_try

ccy_final=[]
outcome_final=[]
avrg_incr_mean_final=[]
last_price=[]
#len(model_final)
for seq_num in range(len(model_final)):
    #seq_num=41
    n= 10
    N= 25
    n_pca= pca_final[seq_num]
    num_kNN= num_kNN_final[seq_num]
    ccy= all_ccy[seq_num][0:6]
    file_path= master_path + '\\' + ccy + '.csv'
    
    
    #1. Step 1: Import data
    ccy_dat= pd.read_csv(file_path)
    ccy_price= ccy_dat['Price']
    
    #2. n- period(Shorter duration) SMA vs N- period (Longer duration) SMA
    # =============================================================================
    # n= 10
    # N= 25
    # =============================================================================
    
    sma_10d, sma_25d= long_short_SMA(ccy_price,n,N)
    
    ccy_price1= pd.concat([ccy_price,sma_10d,sma_25d],axis=1)
    ccy_price1['LS_SMA']=ccy_price1['SMA_Long']- ccy_price1['SMA_Short']   
    ccy_price2= ccy_price1[N:]
    ccy_price2.index= range(len(ccy_price2))
        
    #3. n- period(Shorter duration) EMA vs N- period (Longer duration) EMA
    # =============================================================================
    # Run 2. to get sma_10d, sma_25d
    #n= 10
    #N= 25
    # =============================================================================
    short_multiplier= 1/(n+1)
    ema_short=[ccy_price2['SMA_Short'].values.tolist()[0]]
    #ema_short[0]
    for i in range(1,len(ccy_price2)):
        ema_short.append((ccy_price2['Price'].values.tolist()[i] - ema_short[i-1])*short_multiplier+ \
                         ema_short[i-1])
        
    ema_short= pd.DataFrame(ema_short)
    ema_short.columns= ['EMA_Short']
        
    long_multiplier= 1/(N+1)
    ema_long=[ccy_price2['SMA_Long'].values.tolist()[0]]
    
    for i in range(1,len(ccy_price2)):
        ema_long.append((ccy_price2['Price'].values.tolist()[i]- ema_long[i-1])*long_multiplier+ \
                         ema_long[i-1])
        
    ema_long= pd.DataFrame(ema_long)
    ema_long.columns= ['EMA_Long']
    
    ccy_price2= pd.concat([ccy_price2,ema_short,ema_long],axis=1)
    ccy_price2['LS_EMA']=ccy_price2['EMA_Long']- ccy_price2['EMA_Short']   
    
    #4. Bollinger Bands: N- day STD 
    price_sd= []
    for i in range(N,len(ccy_price)):
        price_sd.append(np.std(ccy_price[i- N:i]))
        
    price_sd= pd.DataFrame(price_sd)
    price_sd.columns= ['Long_STD']
    
    ccy_price3= pd.concat([ccy_price2,price_sd],axis=1)    
    ccy_price3['Low_Band']= ccy_price3['SMA_Long'] - 2*ccy_price3['Long_STD']
    ccy_price3['High_Band']= ccy_price3['SMA_Long'] + 2*ccy_price3['Long_STD']
    ccy_price3['Price-Low']= ccy_price3['Price']- ccy_price3['Low_Band']
    ccy_price3['High-Price']= ccy_price3['High_Band']- ccy_price3['Price']
    ccy_price3['Price_Lag']= ccy_price3['Price'].shift(1)
    ccy_price3['Price_Increment']= ccy_price3['Price']- ccy_price3['Price_Lag']
    
    #5. n- day RSI 
    
    lag_pr= ccy_price.shift(1)
    pr_incr= ccy_price- lag_pr
    pr_incr= pd.DataFrame(pr_incr[1:])
    pr_incr.index= range(len(pr_incr))
    
    pos_RS=[]
    for index, value in enumerate(pr_incr['Price']):
        if pr_incr['Price'][index]>=0:
            pos_RS.append(pr_incr['Price'][index])
        else:
            pos_RS.append(np.float64(0))
            
    neg_RS=[]
    for index, value in enumerate(pr_incr['Price']):
        if pr_incr['Price'][index]<0:
            neg_RS.append(-1*pr_incr['Price'][index])
        else:
            neg_RS.append(np.float64(0))
            
    pos_RS1=[]
    pos_RS_initvect= pos_RS[0:n]
    
    for i in range(n,len(pos_RS)):
    
        if i==n:
            pos_RS1.append(np.mean(pos_RS_initvect))
        else:
            pos_RS1.append((pos_RS1[len(pos_RS1)-2]*(n-1) + pos_RS[i])/np.float64(n))
          
    
    neg_RS1=[]
    neg_RS_initvect= neg_RS[0:n]
    for i in range(n,len(neg_RS)):
        
        if i==n:
            neg_RS1.append(np.mean(neg_RS_initvect))
        else:
            neg_RS1.append((neg_RS1[len(neg_RS1)-1]*(n-1) + neg_RS[i])/np.float64(n))
            
    RS= np.array(pos_RS1)/np.array(neg_RS1)
    RSI= pd.DataFrame(1- 1/(1+RS))
    RSI.columns=['RSI']
    RSI_trim= RSI[len(RSI)-len(ccy_price3):len(RSI)]
    RSI_trim.index=range(len(RSI_trim))
    ccy_price3['RSI']= RSI_trim
    
    #6. Flag increment/ decrement of FX rate
    
    pr_incr= ccy_price3['Price_Increment'].values.tolist()
    pr_flag=[]
    for x in pr_incr:
        if x>0:
            pr_flag.append(1)
        else:
            pr_flag.append(0)
    pr_flag= pd.DataFrame(pr_flag)
    pr_flag.columns= ['Price_Ind']
    ccy_price4= pd.concat([ccy_price3,pr_flag],axis=1)
    ccy_price4['Price_Ind_ld']= ccy_price4['Price_Ind'].shift(-1)
    ccy_price4= ccy_price4[0:len(ccy_price4)-1]
    
    ccy_price_fin= ccy_price4[['LS_SMA','LS_EMA','Price-Low','High-Price','RSI','Price_Increment','Price_Ind_ld']]
    ccy_price_fin= ccy_price_fin[1:len(ccy_price_fin)]
    ccy_price_fin.index= range(len(ccy_price_fin))

    X= ccy_price_fin[['LS_SMA','LS_EMA','Price-Low','High-Price','RSI']]
    X_norm= StandardScaler().fit(X)
    X= pd.DataFrame(X_norm.transform(X))
    X.columns= ['LS_SMA','LS_EMA','Price-Low','High-Price','RSI']
    Y= pd.DataFrame(ccy_price_fin['Price_Ind_ld'])  
    
    pca= PCA(n_components= n_pca)
    X_pca= pca.fit(X)
    X_trans= pd.DataFrame(X_pca.transform(X))
    
    clf3= neighbors.KNeighborsClassifier(n_neighbors= num_kNN)
    clf3.fit(np.array(X_trans),np.array(Y.values.ravel()))
    
    X_pred= pd.DataFrame(ccy_price3.loc[len(ccy_price3)-1,:])
    X_pred.columns= ['Tech_Ind']
    X_pred_ind= pd.DataFrame(X_pred['Tech_Ind'].values.tolist()).T
    X_pred_ind.columns=ccy_price3.columns
    X_last= X_pred_ind['Price']
    X_pred_fin= X_pred_ind[['LS_SMA','LS_EMA','Price-Low','High-Price','RSI']]
    X_pred_fin= pd.DataFrame(X_norm.transform(X_pred_fin))
    X_pred_fin.columns= ['LS_SMA','LS_EMA','Price-Low','High-Price','RSI']
    X_pred_trans= pd.DataFrame(X_pca.transform(X_pred_fin))
    outcome= clf3.predict(np.array(X_pred_trans))
    NNeigh= pd.DataFrame(clf3.kneighbors(X_pred_trans,n_neighbors=num_kNN)[1]).T
    NNeigh.columns=['Index']
    
    list_Neigh=[]

    for neigh in range(len(NNeigh)):
        list_Neigh.append(NNeigh['Index'][neigh])
        
    ori_price_dat= ccy_price4[:]
    ori_price_dat['Index']=ori_price_dat.index
    
    
    in_neigh= ori_price_dat['Index'].isin(list_Neigh)
    in_outcome= ori_price_dat['Price_Ind_ld']== int(outcome)
    ori_price_dat['Dummy1']= in_neigh
    ori_price_dat['Dummy2']= in_outcome
    ori_price_dat['Price_Increment_ld']= ori_price_dat['Price_Increment'].shift(-1)
    
    incr_mean=[]
    for index_neigh in range(len(ori_price_dat)):
        
        if ori_price_dat['Dummy1'][index_neigh]==True and ori_price_dat['Dummy2'][index_neigh]==True :
            incr_mean.append(ori_price_dat['Price_Increment_ld'][index_neigh])
    
    avrg_incr_mean=np.mean(incr_mean)
    
    ccy_final.append(ccy)
    outcome_final.append(outcome)
    avrg_incr_mean_final.append(avrg_incr_mean)
    last_price.append(X_last[0])

export_ML_path= r'C:\Users\jason\OneDrive\Documents\Quantopian\FX Trading\G10 Historical Rates\Result'
result_summary= pd.concat([pd.DataFrame(ccy_final),pd.DataFrame(outcome_final),pd.DataFrame(avrg_incr_mean_final),pd.DataFrame(last_price)],axis=1)
result_summary.columns=['ccy_pair','signal','increment','Price']
result_summary['New_price']= result_summary['Price']+ result_summary['increment']
result_summary.to_csv(export_ML_path + '\\result_summary.csv')

# =============================================================================
# ccy_final= ccy_final[0:len(ccy_final)-1]
# outcome_final= outcome_final[0:len(outcome_final)-1]
# avrg_incr_mean_final=avrg_incr_mean_final[0:len(avrg_incr_mean_final)-1]
# last_price=last_price[0:len(last_price)-1]
# =============================================================================

######################## OANDA API ####################################
client = API(access_token=access_token)

params= {'to': '2018-01-02T00:00:00Z', \
         'from': '2018-12-31T08:00:00Z', 'granularity': 'D1'}

instr= 'AUD_USD'

def cnv(r, h):
    for candle in r.get('candles'):
        ctime = candle.get('time')[0:19]
        try:
            rec = "{time},{complete},{o},{h},{l},{c},{v}".format(
                time=ctime,
                complete=candle['complete'],
                o=candle['mid']['o'],
                h=candle['mid']['h'],
                l=candle['mid']['l'],
                c=candle['mid']['c'],
                v=candle['volume'],
            )
        except Exception as e:
            print(e, r)
        else:
            h.write(rec+"\n")

temp_path= r'C:\Users\jason\OneDrive\Documents\Quantopian\FX Trading\G10 Historical Rates\Processed Data\Result'
with open(temp_path + "\\{}.{}.csv".format(instr,'H4'), "w") as O:           
    for r in InstrumentsCandlesFactory(instrument=instr, params=params):
        rv = client.request(r)
        cnv(r.response, O)
        
ccy_import= pd.read_csv(temp_path + "\\" + instr_label + ".csv")
ccy_import.columns=['Date','Logic','Open','High','Low','Price','Volume']
ccy_import.to_csv(temp_path + "\\" + instr_label + ".csv")

