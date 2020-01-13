# For not showing warnings in terminal
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

print("There are many data whose Year is not given")
print("So in this project we will use regression algorithm to guess year of landing on the basis of coordinates of the meteorites")

#reading the data
df = pd.read_csv('meteorite-landings.csv')

# Showing dataframe Info:
df.info()

# Removing the rows that have Not Valid "latitude" and "longitude" values:
df = df[(df["reclat"] != 0.0) & (df["reclong"] != 0.0)]

#preparing the data
df1 = df[['reclat','reclong','year']]
df1_clean = df1.dropna()

X= np.array(df1_clean.as_matrix(columns=['reclat','reclong']))
y= np.array(df1_clean.as_matrix(columns=['year']).ravel())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#Setting up the BaggingClassifier
#It is used here for its spead (as it compared the data only within a certain range)
reg = BaggingClassifier(KNeighborsClassifier(n_neighbors=50,n_jobs=3),max_samples=0.5, max_features=0.5)
reg.fit(X_train,y_train)
print("The accuracy is: ",reg.score(X_test,y_test))
pred=reg.predict(X_test)
print("Some examples of the guessed years, and the correct ones:")
print(pred[10:])
print(y_test[10:])