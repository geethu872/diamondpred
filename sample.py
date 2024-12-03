import pandas as pd
import numpy as np
import pickle
df=pd.read_csv("diamonds.csv")
print(df.head())
df=df[(df['x']!=0) & df['y']!=0 & (df['z']!=0)]
df1=df.copy()
def detect_outliers(df1, columns):
    outliers = {}
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[column] = df1[(df1[column] < lower_bound) | (df1[column] > upper_bound)][column]
    return outliers
columns_to_check = ['x', 'y', 'z', 'depth', 'table']
outliers = detect_outliers(df1, columns_to_check)
outliers_count = sum(len(outlier_values) for outlier_values in outliers.values())
outliers_count
outlier_indices = set()
for outlier_values in outliers.values():
    outlier_indices.update(outlier_values.index)
df1= df1.drop(index=outlier_indices)
df1=df1.drop('Unnamed: 0',axis=1)
from sklearn.preprocessing import LabelEncoder
# li = ["cut","color","clarity"]
# label = LabelEncoder()
# for i in li:
#     df1[i] = label.fit_transform(df1[i])
from sklearn.preprocessing import LabelEncoder
label1 = LabelEncoder()
df1['clarity']=label1.fit_transform(df1['clarity'])
label2 = LabelEncoder()
df1['cut']=label2.fit_transform(df1['cut'])
label3 = LabelEncoder()
df1['color']=label3.fit_transform(df1['color'])

df1.head(10)
x=df1.drop('price',axis=1)
y=df1['price']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
xtrain=pd.DataFrame(scale.fit_transform(xtrain))
xtest=pd.DataFrame(scale.transform(xtest))
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import XGBRegressor
pipeline=Pipeline([
    ('clf',xgb.XGBRegressor())
])
param_grid = {
    'clf__learning_rate': [0.01, 0.1, 0.3],  
    'clf__n_estimators': [100, 300, 500],  
    'clf__max_depth': [3, 5, 7],                
    'clf__gamma': [0, 0.1, 0.3],            
    'clf__subsample': [0.8, 1.0]           
          
}
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, 
                           cv=3, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
grid_search.fit(xtrain,ytrain)
best_params=grid_search.best_params_
best_learning_rate = best_params['clf__learning_rate']
best_n_estimators = best_params['clf__n_estimators']
best_max_depth = best_params['clf__max_depth']
best_gamma = best_params['clf__gamma']
best_subsample = best_params['clf__subsample']
model2=xgb.XGBRegressor(learning_rate=best_learning_rate,
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    gamma=best_gamma,
    subsample=best_subsample)
model2.fit(xtrain,ytrain)
prediction1=model2.predict(xtest)
mae=mean_absolute_error(ytest,prediction1)
r2score=model2.score(xtest,ytest)
r2_train=model2.score(xtrain,ytrain)
print(f"Test MAE: {mae}")
print('R-squared-train',r2_train)
print(f"R-squared: {r2score}")

pickle.dump(model2,open('xgb.pkl','wb'))

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scale, f)

with open('label1.pkl', 'wb') as f:
    pickle.dump(label1, f)

with open('label2.pkl', 'wb') as f:
    pickle.dump(label2, f)

with open('label3.pkl', 'wb') as f:
    pickle.dump(label3, f)