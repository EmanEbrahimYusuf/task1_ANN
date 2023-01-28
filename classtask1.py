from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import seaborn as sns
import random
 
class TaskOne:
 
    def __init__(self,data,columsList,lable1,lable2) -> None:
        self.weights=np.random.rand(len(columsList))*0.01
        self.bios=0
        self.columsList=columsList
        self.lable1=lable1
        self.lable2=lable2
        self.data=data
        self.columsList2=columsList
       # self.columsList2.append("species")
        self.trainData,self.testData=self.get_data()
 
        print(len(self.trainData),len(self.testData))
 
    def get_data(self):
        d1=self.data[self.data['species']==self.lable1]
        d1trian,d1test=d1[:30],d1[30:50]
        d2=self.data[self.data['species']==self.lable2]
        d2trian,d2test=d2[:30],d2[30:50]
        datatrain=shuffle(pd.concat([d1trian,d2trian]))
        datatest=shuffle(pd.concat([d1test,d2test]))
 
        return datatrain,datatest
 
    def signum(self,net):
        if net>0:
            return 1
        elif net<0:
            return -1
        elif net == 0:
            return 0 
 
    def lable(self,x):
            if x==self.lable1:
                return 1
            else: return -1
 
    def fit(self,learnRate,epochs,biosed=True):
 
        self.trainData['species']=self.trainData['species'].apply(self.lable)
        self.trainData['species']=self.trainData['species'].astype('int64')
        for i in range(epochs):
            for X,Y in zip(self.trainData[self.columsList].values,self.trainData['species'].values):
                net=np.dot(self.weights.T,np.array(X))+self.bios
                Q=self.signum(net)
                loss=Y-Q
                self.weights=self.weights+learnRate*loss*X
                if biosed:
                    self.bios=self.bios+learnRate*loss
        return self.weights,self.bios
    
    # def fitAdaline(self,learnRate,epochs,biosed=True):
 
    #     self.trainData['species']=self.trainData['species'].apply(self.lable)
    #     self.trainData['species']=self.trainData['species'].astype('int64')
    #     for i in range(epochs):
    #         X=np.array(self.trainData[self.columsList].values)
    #         Y=np.array(self.trainData['species'].values)
    #         y_pred=np.dot(X,self.weights)+self.bios
    #         loss=Y-y_pred
    #         db=(np.sum(y_pred-Y))/len(Y)
    #         dw=(np.dot((y_pred-Y),X))/len(Y)


    #         self.weights=self.weights+learnRate*dw
    #         self.bios=self.bios+learnRate*db
    #         # for X,Y in zip(self.trainData[self.columsList].values,self.trainData['species'].values):
    #         #     net=np.dot(self.weights.T,np.array(X))+self.bios
    #         #     #Q=self.signum(net)
    #         #     loss=Y-net
    #         #     self.weights=self.weights+learnRate*loss*X
    #         #     if biosed:
    #         #         self.bios=self.bios+learnRate*loss
    #         #     mse = (((net - Y)**2).sum())/len(y) 
    #     return self.weights,self.bios
 
 
    def predict(self):
        ydash=[]
        def lable(x):
            if x==self.lable1:
                return 1
            else: return -1
        self.testData['species']=self.testData['species'].apply(lable)
        self.testData['species']=self.testData['species'].astype('int64')
        for X in self.testData[self.columsList].values:
                net=np.dot(self.weights.T,np.array(X))+self.bios
                ydash.append(self.signum(net))
 
        return np.array(ydash)
 
    def score(self):
        self.pred=self.predict()
        sumCorrectItems=0
        for i,j in zip(self.pred,self.testData['species']):
            if i==j:
                sumCorrectItems+=1
        return sumCorrectItems/len(self.testData)
 
    def draw(self):
        np.random.seed(19680801)
        df1=self.testData[self.testData["species"]==1]
        df2=self.testData[self.testData["species"]==-1]
        plt.scatter(df1[self.columsList[0]],df1[self.columsList[1]],c='red')
        plt.scatter(df2[self.columsList[0]],df2[self.columsList[1]],c='blue')
        x1_1=int(min(self.testData[self.columsList[0]]))
        x1_2=(-self.bios-(x1_1*self.weights[0]))/self.weights[1]
 
        x2_1=int(max(self.testData[self.columsList[0]]))
        x2_2=(-self.bios-(x2_1*self.weights[0]))/self.weights[1]
 
        plt.plot([x1_1,x2_1],[x1_2,x2_2])
 
        plt.xlabel(self.columsList[0])
        plt.ylabel(self.columsList[1])
 
        # naming the title of the plot
        plt.title("Plot between {0} & {1}".format(self.columsList[0],self.columsList[1]))
        plt.show()
 
 
    def confusionMatrix(self):
        TP=0
        FP=0
        FN=0
        TN=0
 
        for i in range(len(self.testData)):
            if(self.pred[i]==1 and self.testData['species'].values[i]==1):
                TP+=1
            elif(self.pred[i]==-1 and self.testData['species'].values[i]==-1):
                TN+=1
            elif(self.pred[i]==1 and self.testData['species'].values[i]==-1):
                FP+=1
            elif(self.pred[i]==-1 and self.testData['species'].values[i]==1):
                FN+=1
 
        return np.array([[TP,FP],[FN,TN]])
 
 
    def line(self):
        x1_1=random.randint(1,20)
        x1_2=(-self.bios-(x1_1*self.weights[0]))/self.weights[1]
 
        x2_1=random.randint(5,25)
        x2_2=(-self.bios-(x2_1*self.weights[0]))/self.weights[1]
        #plt.plot([x1_1,x2_1],[x1_2,x2_2])
        #plt.show()
        return [(x1_1,x1_2),(x2_1,x2_2)]


def task(columsList,lable1,lable2,L,epochs,bias):

  data =pd.read_csv('penguins.csv')
  data['gender'].fillna('male',inplace=True)

  def genderlable(x):
    if x=='male':
      return 1
    else :return 0
  data['gender']=data['gender'].apply(genderlable)
  model=TaskOne(data,columsList,lable1,lable2)
  #model=TaskOne(data,['bill_depth_mm', 'flipper_length_mm'],'Adelie','Chinstrap')
  model.fitAdaline(L,epochs,bias)
  print(model.score())
  print(model.confusionMatrix())
  model.draw()
#task(['flipper_length_mm', 'bill_depth_mm'],'Adelie','Chinstrap',0.01,4000,True)


#task(['bill_depth_mm', 'flipper_length_mm'],'Adelie','Chinstrap',0.1,200,False)