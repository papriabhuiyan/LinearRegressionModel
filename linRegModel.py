import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

housing_data = pd.read_csv('housing_data.csv')
X = housing_data.drop(['No', 'Y price per sqft'], axis=1)
X.replace({'condition':{'Poor':0, 'Average':1, 'Good':2, 'Excellent':3}}, inplace=True)
Y = housing_data['Y price per sqft']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)
print("Linear Regression Model created")
training_data_prediction = lin_reg_model.predict(X_train)
training_data_prediction = lin_reg_model.predict(X_train)
train_error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error for the Training Set: ", train_error_score)


Y_predictions = lin_reg_model.predict(X_test)
test_error_score = metrics.r2_score(Y_test, Y_predictions)
print("R squared Error for the Testing Set: ", test_error_score)
sns.regplot(x=Y_test, y=Y_predictions, scatter_kws={"color": "pink"}, line_kws={"color": "green"})