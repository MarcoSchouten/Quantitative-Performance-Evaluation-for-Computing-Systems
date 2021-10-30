from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


import statsmodels.api as sm
from statsmodels.formula.api import ols



# read the data and creat the dataset
df = pd.read_csv("bank-full.csv", delimiter=";")
encoded = df.apply(LabelEncoder().fit_transform)

x = encoded.iloc[:, :16]
y = encoded.iloc[:, 16]

# define the model
# cores=jobs= 1,2,8
# estimators = 100, 200, 400, 500.
f = open('script_output.csv', 'w')
writer = csv.writer(f)

myjobs= [1, 2, 8]
myestimators= [100, 200, 400, 500]
for j in range(0, len(myjobs)):
    for e in range(0, len(myestimators)):
        for t in range(0, 3):
            model = RandomForestClassifier(n_estimators = myestimators[e] , n_jobs = myjobs[j])
            start = time()

            # perform search
            model.fit(x, y)

            # record current time
            end = time()

            # report execution time
            result = end - start
            # result = np.around(result, decimals=3, out=None)
            # print(np.around(result, decimals=3, out=None))

            row ="{};{};{:.3f}".format(myestimators[e], myjobs[j], result)
            print(row)
            writer.writerow(row)
