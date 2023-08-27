import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('Salary Data.csv')

mean_age = df['Age'].mean()
year_exp = df['Years of Experience'].mean()
mean_salary = df['Salary'].mean()
df['Age'] = df['Age'].fillna(mean_age)
df['Years of Experience'] = df['Years of Experience'].fillna(year_exp)
df['Salary'] = df['Salary'].fillna(mean_salary)

df = df.dropna()
df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1})
df.drop(columns = ['Education Level', 'Job Title'], inplace = True)
X = df.drop(columns = ['Salary'])
y = df['Salary']
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

rf = RandomForestRegressor()
xg = XGBRegressor()

rf.fit(X_train, y_train)
xg.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_xg = xg.predict(X_test)

train_accuracy_rf = rf.score(X_train,y_train)
test_accuracy_rf = rf.score(X_test, y_test)

print('train_accuracy_rf:', train_accuracy_rf)
print('test_accuracy_rf:', test_accuracy_rf)

train_accuracy_xg = xg.score(X_train,y_train)
test_accuracy_xg = xg.score(X_test, y_test)

print('train_accuracy_xg:', train_accuracy_xg)
print('test_accuracy_xg:', test_accuracy_xg)


mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)

print('mae:', mae)
print('mse:', mse)


# pickle both the models
import pickle
pickle.dump(rf, open('model_rf.pkl', 'wb'))
pickle.dump(xg, open('model_xg.pkl', 'wb'))

#Xg is better than rf