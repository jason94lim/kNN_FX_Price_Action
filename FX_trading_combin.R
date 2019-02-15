library(utils)
library(sqldf)
library(readr)
library(dplyr)

dat_extract<- function(master_path){
  setwd(master_path)
  file_list<- list.files(path=master_path,pattern=".csv")
  file_list_1<- gsub(" ","",file_list)
  flie_list_2<- gsub(".csv","",file_list_1)
  
  for (i in 1:length(file_list)){
    in_file<- file_list[i]
    out_file<- flie_list_2[i]
    file<- paste0(master_path,"/",in_file)
    assign(out_file,read_csv(file,col_types = cols(Date = col_date(format = "%d/%m/%Y"))),envir = globalenv())
  }
}

g10_ccy<- c('EUR','GBP','USD','CAD','SEK','CHF','NOK','JPY','AUD','NZD')
g10_combin<- combn(g10_ccy,2)
g10_combin_t<- as.data.frame(t(g10_combin))
names(g10_combin_t)<- c('CCY1','CCY2')
g10_combin_t$'G10_ccy'<- paste0(g10_combin_t$CCY1,'\\',g10_combin_t$CCY2)
g10_combin_t2<- g10_combin_t[(g10_combin_t$CCY1!='USD'),]
g10_combin_t3<- g10_combin_t2[(g10_combin_t2$CCY2!='USD'),]

hist_rate_path<- 'C:/Users/jason/OneDrive/Documents/Quantopian/FX Trading/G10 Historical Rates'
dat_extract(hist_rate_path)

#Arrange for CCY1
test_ccy_pair<-NULL
id_ccy1<- NULL
for (i in 1:nrow(g10_combin_t3)){
  if (as.character(g10_combin_t3$CCY1[i]) %in% c("AUD","EUR","GBP","NZD")){
    test<- paste0(as.character(g10_combin_t3$CCY1[i]),"USD")
    id<- 1
  }else{
    test<- paste0("USD",as.character(g10_combin_t3$CCY1[i]))
    id<- 0
  }
  test_ccy_pair<- rbind(test_ccy_pair,test)
  id_ccy1<- rbind(id_ccy1,id)
}

#Arrange for CCY2
test_ccy_pair2<-NULL
id_ccy2<-NULL
for (i in 1:nrow(g10_combin_t3)){
  if (as.character(g10_combin_t3$CCY2[i]) %in% c("AUD","EUR","GBP","NZD")){
    test2<- paste0(as.character(g10_combin_t3$CCY2[i]),"USD")
    id2<-1
  }else{
    test2<- paste0("USD",as.character(g10_combin_t3$CCY2[i]))
    id2<-0
  }
  test_ccy_pair2<- rbind(test_ccy_pair2,test2)
  id_ccy2<-rbind(id_ccy2,id2)
}

#Combine all data sets
g10_combin_t3['CCY_Pair_1']<- test_ccy_pair
g10_combin_t3['CCY_Pair_2']<- test_ccy_pair2
g10_combin_t3['CCY_ID_1']<- id_ccy1
g10_combin_t3['CCY_ID_2']<- id_ccy2

for (i in 1: nrow(g10_combin_t3)){
  ccy1<- as.character(g10_combin_t3['CCY_Pair_1'][i,])
  ccy_id1<- as.numeric(g10_combin_t3['CCY_ID_1'][i,])
  ccy2<- as.character(g10_combin_t3['CCY_Pair_2'][i,])
  ccy_id2<- as.numeric(g10_combin_t3['CCY_ID_2'][i,])
  
  if (ccy_id1==1 && ccy_id2==1){
    ccy1_dat<- get(ccy1)[c('Date','Price')]
    names(ccy1_dat)<- c("Date","ccy1")
    ccy2_dat<- get(ccy2)[c('Date','Price')]
    ccy2_dat$Price<- (ccy2_dat$Price)^(-1)
    names(ccy2_dat)<- c("Date","ccy2")
    sql_stat<- paste0("SELECT a.Date, a.ccy1, b.ccy2",
                      " FROM ccy1_dat as a LEFT JOIN ccy2_dat as b",
                      " ON a.Date=b.Date",
                      " ORDER BY a.Date")
    ccy_combin_dat<-sqldf(sql_stat)
    ccy_combin_temp<- as.numeric(ccy_combin_dat$ccy1)*as.numeric(ccy_combin_dat$ccy2)
    ccy_combin_dat$ccy3<- ccy_combin_temp
    ccy_combin_dat<- ccy_combin_dat[!is.na(ccy_combin_dat$ccy3),]
    ccy_combin_dat<- ccy_combin_dat[,c('Date','ccy3')]
    ccy_combin_name<- paste0(substr(ccy1,1,3),substr(ccy2,1,3))
    names(ccy_combin_dat)<- c("Date","Price")
  }else if (ccy_id1==1 && ccy_id2==0){
    ccy1_dat<- get(ccy1)[c('Date','Price')]
    names(ccy1_dat)<- c("Date","ccy1")
    ccy2_dat<- get(ccy2)[c('Date','Price')]
    ccy2_dat$Price<- ccy2_dat$Price
    names(ccy2_dat)<- c("Date","ccy2")
    sql_stat<- paste0("SELECT a.Date, a.ccy1, b.ccy2",
                      " FROM ccy1_dat as a LEFT JOIN ccy2_dat as b",
                      " ON a.Date=b.Date",
                      " ORDER BY a.Date")
    ccy_combin_dat<-sqldf(sql_stat)
    ccy_combin_temp<- as.numeric(ccy_combin_dat$ccy1)*as.numeric(ccy_combin_dat$ccy2)
    ccy_combin_dat$ccy3<- ccy_combin_temp
    ccy_combin_dat<- ccy_combin_dat[!is.na(ccy_combin_dat$ccy3),]
    ccy_combin_dat<- ccy_combin_dat[,c('Date','ccy3')]
    ccy_combin_name<- paste0(substr(ccy1,1,3),substr(ccy2,4,6))
    names(ccy_combin_dat)<- c("Date","Price")
  }else if (ccy_id1==0 && ccy_id2==1){
    ccy1_dat<- get(ccy1)[c('Date','Price')]
    ccy1_dat$Price<- (ccy1_dat$Price)^(-1)
    names(ccy1_dat)<- c("Date","ccy1")
    ccy2_dat<- get(ccy2)[c('Date','Price')]
    ccy2_dat$Price<- (ccy2_dat$Price)^(-1)
    names(ccy2_dat)<- c("Date","ccy2")
    sql_stat<- paste0("SELECT a.Date, a.ccy1, b.ccy2",
                      " FROM ccy1_dat as a LEFT JOIN ccy2_dat as b",
                      " ON a.Date=b.Date",
                      " ORDER BY a.Date")
    ccy_combin_dat<-sqldf(sql_stat)
    ccy_combin_temp<- as.numeric(ccy_combin_dat$ccy1)*as.numeric(ccy_combin_dat$ccy2)
    ccy_combin_dat$ccy3<- ccy_combin_temp
    ccy_combin_dat<- ccy_combin_dat[!is.na(ccy_combin_dat$ccy3),]
    ccy_combin_dat<- ccy_combin_dat[,c('Date','ccy3')]
    ccy_combin_name<- paste0(substr(ccy1,4,6),substr(ccy2,1,3))
    names(ccy_combin_dat)<- c("Date","Price")
  }else{
    ccy1_dat<- get(ccy1)[c('Date','Price')]
    ccy1_dat$Price<- (ccy1_dat$Price)^(-1)
    names(ccy1_dat)<- c("Date","ccy1")
    ccy2_dat<- get(ccy2)[c('Date','Price')]
    ccy2_dat$Price<- ccy2_dat$Price
    names(ccy2_dat)<- c("Date","ccy2")
    sql_stat<- paste0("SELECT a.Date, a.ccy1, b.ccy2",
                      " FROM ccy1_dat as a LEFT JOIN ccy2_dat as b",
                      " ON a.Date=b.Date",
                      " ORDER BY a.Date")
    ccy_combin_dat<-sqldf(sql_stat)
    ccy_combin_temp<- as.numeric(ccy_combin_dat$ccy1)*as.numeric(ccy_combin_dat$ccy2)
    ccy_combin_dat$ccy3<- ccy_combin_temp
    ccy_combin_dat<- ccy_combin_dat[!is.na(ccy_combin_dat$ccy3),]
    ccy_combin_dat<- ccy_combin_dat[,c('Date','ccy3')]
    ccy_combin_name<- paste0(substr(ccy1,4,6),substr(ccy2,4,6))
    names(ccy_combin_dat)<- c("Date","Price")
    
  }
  
  write.csv(ccy_combin_dat,paste0(hist_rate_path,"/Processed Data/",ccy_combin_name,".csv"))
  
}




