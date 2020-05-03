import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as SI
from sklearn.preprocessing import  LabelBinarizer as LB
from sklearn.preprocessing import StandardScaler as SS
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv("dia1.csv")
not_req=['Unnamed: 0']
#clean df
df=df.drop(not_req,axis=1)
#prepare target 
target='price'
y=df[target]
#prepare predictors
X=df.drop(target,axis=1)
#train ,test split
X_train,X_test,y_train,y_test=train_test_split(X,y)

#create a mapper
mapper=DataFrameMapper([(['cut'],[SI(strategy='most_frequent'),LB()]),
(['color'],[SI(strategy='most_frequent'),LB()]),
(['clarity'],[SI(strategy='most_frequent'),LB()]),
(['carat'],[SI(strategy='mean'),SS()]),],df_out=True)

#for the mapper and transform the feature
Z_train=mapper.fit_transform(X_train)
Z_test=mapper.transform(X_test)

#create a model
model=LinearRegression()
model.fit(Z_train,y_train)
#print(model.score(Z_test,y_test))

#create a pipeline

pipe=make_pipeline(mapper,model)
#print(pipe.fit(X_train,y_train))

#test the model
new=pd.DataFrame(
    {
        'carat':[2],
        'cut':['Ideal'],
        'color':['E'],
        'clarity':['VVS1'],
    }
)
print("predicted value")
print(pipe.predict(new))
'''
#dump the python object
with open("model/pipe.pkl","wb") as f:
    pickle.dump(pipe,f)

del pipe
with open("model/pipe.pkl","rb") as f:
    pipe=pickle.load(f)

print(pipe.predict(new)[0])'''

with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print('Pickle Dumped!')

del pipe

with open('pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)

print('Loaded!')


print(pipe.predict(new)[0])