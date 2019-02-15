# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 00:08:18 2019

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


#0. Configuration
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

for seq_num in range(len(all_ccy)):
    n= 10
    N= 25
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
    ema_short[0]
    for i in range(1,len(ccy_price2)-1):
        ema_short.append((ccy_price2['Price'].values.tolist()[i] - ema_short[i-1])*short_multiplier+ \
                         ema_short[i-1])
        
    ema_short= pd.DataFrame(ema_short)
    ema_short.columns= ['EMA_Short']
        
    long_multiplier= 1/(N+1)
    ema_long=[ccy_price2['SMA_Long'].values.tolist()[0]]
    
    for i in range(1,len(ccy_price2)-1):
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
    
    #7.kNN fitting: Non- normalised, Non- PCA
    ccy_price_fin= ccy_price4[['LS_SMA','LS_EMA','Price-Low','High-Price','RSI','Price_Increment','Price_Ind_ld']]
    ccy_price_fin= ccy_price_fin[1:len(ccy_price_fin)]
    ccy_price_fin.index= range(len(ccy_price_fin))
    # =============================================================================
    # X= ccy_price_fin[['LS_SMA','LS_EMA','Price-Low','High-Price','RSI']]
    # Y= pd.DataFrame(ccy_price_fin['Price_Ind_ld'])
    # 
    # X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2)
    # 
    # 
    # kNN=[]
    # accur=[]
    # for n in range(1,300):
    #     clf= neighbors.KNeighborsClassifier(n_neighbors= n)
    #     clf.fit(np.array(X_train),np.array(Y_train.values.ravel()))
    #     accuracy= clf.score(X_test,Y_test)
    #     kNN.append(n)
    #     accur.append(accuracy)
    # accur_table= pd.concat([pd.DataFrame(kNN),pd.DataFrame(accur)],axis=1)
    # =============================================================================
    # =============================================================================
    # 
    # accuracy= clf.score(X_test,Y_test)
    # predict_prob= clf.predict_proba(X_test)
    # kNNeighbors= clf.kneighbors(X_test,n_neighbors=n_kNN)
    # kNN_model= clf
    # 
    # =============================================================================
    
    #7.kNN fitting: Normalised, Non- PCA
    # =============================================================================
    # X= ccy_price_fin[['LS_SMA','LS_EMA','Price-Low','High-Price','RSI']]
    # X_norm= StandardScaler().fit(X)
    # X= pd.DataFrame(X_norm.transform(X))
    # X.columns= ['LS_SMA','LS_EMA','Price-Low','High-Price','RSI']
    # Y= pd.DataFrame(ccy_price_fin['Price_Ind_ld'])
    # 
    # X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2)
    # 
    # 
    # kNN2=[]
    # accur2=[]
    # for n in range(1,300):
    #     clf2= neighbors.KNeighborsClassifier(n_neighbors= n)
    #     clf2.fit(np.array(X_train),np.array(Y_train.values.ravel()))
    #     accuracy= clf2.score(X_test,Y_test)
    #     kNN2.append(n)
    #     accur2.append(accuracy)
    #     
    # accur_table2= pd.concat([pd.DataFrame(kNN2),pd.DataFrame(accur2)],axis=1)
    # =============================================================================
    
    #8.kNN fitting: Mormalised, PCA
    X= ccy_price_fin[['LS_SMA','LS_EMA','Price-Low','High-Price','RSI']]
    X_norm= StandardScaler().fit(X)
    X= pd.DataFrame(X_norm.transform(X))
    X.columns= ['LS_SMA','LS_EMA','Price-Low','High-Price','RSI']
    Y= pd.DataFrame(ccy_price_fin['Price_Ind_ld'])
    
    X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2)
    
    pca3=[]
    kNN3=[]
    accur3=[]
    for n_pca in range(2,len(X_train.columns)):
        pca= PCA(n_components= n_pca)
        X_train_pca= pca.fit(X_train)
        X_train_trans= pd.DataFrame(X_train_pca.transform(X_train))
        X_test_trans= pd.DataFrame(X_train_pca.transform(X_test))
        
        for n in range(1,300):
            clf3= neighbors.KNeighborsClassifier(n_neighbors= n)
            clf3.fit(np.array(X_train_trans),np.array(Y_train.values.ravel()))
            accuracy= clf3.score(X_test_trans,Y_test)
            pca3.append(n_pca)
            kNN3.append(n)
            accur3.append(accuracy)
            
    accur_table3= pd.concat([pd.DataFrame(pca3),pd.DataFrame(kNN3),pd.DataFrame(accur3)],axis=1)
    accur_table3.columns= ['n_pca','n_kNN','accuracy']
    accur_table3= accur_table3.sort_values(by='accuracy',ascending= False)
    accur_table3.index= range(len(accur_table3))
    final_combin= accur_table3.loc[range(10),:]
    
    export_path= r'C:\Users\jason\OneDrive\Documents\Quantopian\FX Trading\ML Result Table'
    final_combin.to_csv(export_path + '\\' + ccy + '.csv')
