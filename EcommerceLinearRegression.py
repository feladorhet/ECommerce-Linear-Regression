import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#READ IN THE DATA SET
customers = pd.read_csv("Ecommerce Customers.csv")

##DATA 
##EXPLORATION 
##STARTS HERE

#CASUAL CHECKS ON THE SET
#custHead = customers.head()
#custInfo = customers.info()
#custDesc = customers.describe()

#jointplot to compare the "Time on Website" and "Yearly Amount Spent"
timeOnWeb_yearlySpending = sns.jointplot(data=customers, x="Time on Website", y="Yearly Amount Spent")
#plt.plot()

#2D hex bin plot comparing "Time on App" and "Length of Membership"
timeOnApp_membershipLength = sns.jointplot(data=customers, x="Time on App", y="Length of Membership", kind="hex")
#plt.plot()

#relation among data
relGraph = sns.pairplot(customers)
#plt.plot()

#linear model plot of "Yearly Amount Spent" vs. "Length of Membership"
yearlySpending_membershipLength = sns.lmplot(data=customers, x="Length of Membership", y="Yearly Amount Spent")
#plt.plot()

##DATA 
#EXPLORATION 
#ENDS HERE


##NOW TO TRAIN A LINEAR REGRESSION MODEL TO DETERMINE WHAT EFFECTS YEARLY SPENDINGS
X = customers[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]
y = customers["Yearly Amount Spent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 

linearModel = LinearRegression()
linearModel.fit(X_train, y_train)
predictions = linearModel.predict(X_test)

#SCATTER PLOT TO SEE THE PERFORMANCE OF THE MODEL
plt.scatter(y=y_test, x=predictions)
plt.xlabel("Predictions")
plt.ylabel("Test Data")
#plt.show()

#PERFORMANCE METRICS 
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

#HISTOGRAM OF THE RESIDUALS
sns.histplot(data=y_test - predictions, bins=50, kde=True)
#plt.show()

#COEFFICIENTS OF THE FACTORS
coeff = pd.DataFrame(data=linearModel.coef_, index=X.columns, columns=["Coefficient"])
#print(coeff)
#SINCE THE HIGHEST COEFFICIENT BELONGS TO "Length of Membership", IT MAKES SENSE TO ASSUME THAT THE LONGER A CUSTOMER STAYS AS A MEMBER, MORE LIKELY THEY ARE TO SPEND MONEY

