import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('50_Startups.csv')
x=df.iloc[:,:-1].values
y=df.iloc[:,4].values

from sklearn.preprocessing import OneHotEncoder
one_hot_encoded_data = pd.get_dummies(df,columns = ["State"])
x=one_hot_encoded_data.drop(["Profit"],axis=1)
y=one_hot_encoded_data[["Profit"]]

from sklearn.model_selection import train_test_split,cross_val_score
x_train,x_test,y_train,y_test=train_test_split(x,y ,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print("y test",y_test)
print("y pred",y_pred)

cv_scores = cross_val_score(regressor, x, y, cv=3, scoring='r2')

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation R-squared:", np.mean(cv_scores))


from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
print("rmse",rmse)
print("r2",r2)

print('Train Score:', regressor.score(x_train, y_train))
print('Test Score:', regressor.score(x_test, y_test))
