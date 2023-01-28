from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from classtask1 import TaskOne


def task(columsList,lable1,lable2,L,epochs,bios):

    data =pd.read_csv('penguins.csv')
    data['gender'].fillna('male',inplace=True)

    def genderlable(x):
        if x=='male':
            return 1
        else :return 0
    data['gender']=data['gender'].apply(genderlable)
    model=TaskOne(data,columsList,lable1,lable2)
    #model=TaskOne(data,['bill_depth_mm', 'flipper_length_mm'],'Adelie','Chinstrap')
    model.fit(L,epochs,bios)
    print(model.score())
    print(model.confusionMatrix())
    model.draw()
#task(['flipper_length_mm', 'bill_depth_mm'],'Adelie','Chinstrap',0.01,4000,True)