from scipy.stats import loguniform
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd  # used to load the data
import numpy as np  # optimized numerical library
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing, metrics, utils, decomposition, model_selection, linear_model, discriminant_analysis, svm, tree, ensemble
# library providing several ML algorithms and related utility
from collections import Counter

# ML MODELS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor









if __name__ == '__main__':
    # _______________________________________________ LOAD DATA _______________________________________________
    # Get the dataset loaded and define class labels
    data = pd.read_csv('response_time.csv', header=0)
    # print(data.head())

    # All data columns except last are input features (X), last column is output label (y)
    n_features = len(data.columns) - 1
    X = data.iloc[:, 0:n_features].values
    y = data.iloc[:, n_features].values

    # Scale data
    # Train a scaler to standardize the features (zero mean and unit variance)
    X = preprocessing.StandardScaler().fit_transform(X)
    # print(np.mean(X_small),np.std(X_small))

    # _______________________________________________ DIMENSIONALITY  REDUCTION ______________________
    # Train a PCA with k dimensions: Write a script to find minimum number of k components that ...
    # contain 90% of variance description
    feat_cols = ['feature' + str(i) for i in range(X.shape[1])]
    small_df = pd.DataFrame(X, columns=feat_cols)
    # print(small_df.head())

    n_components = 3
    pca_small = decomposition.PCA(n_components)
    principalComponents_small = pca_small.fit_transform(X)
    pca_cols = ['pc' + str(i) for i in range(n_components)]
    principal_small_Df = pd.DataFrame(data=principalComponents_small, columns=pca_cols)
    # print(principal_small_Df.tail())
    # print('Explained variation per principal component: {}'.format(pca_small.explained_variance_ratio_))
    sum_Var = np.sum(pca_small.explained_variance_ratio_) * 100
    print('SumVar' + str(np.round(sum_Var, 0)))
    X = principal_small_Df.iloc[:, 0:n_components]

    # _______________________________________________ SPLIT DATA _______________________________________________
    # Split data in training and testing for 0.33 ratio of testing
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
    print('X train {}'.format(X_train.shape))
    print('X test{}'.format(X_test.shape))
    print('y train {}'.format(y_train.shape))
    print('y train {}'.format(y_test.shape))




    # ___________________________________________ ml models to compare against the theory regression






 # ___________________________________________ ml models to compare against the theory regression
    # MODEL CREATION
    MLP = MLPRegressor()
    RF = RandomForestRegressor()
    LR = LinearRegression()

    # _______________________________________________ HYPERPARAMETER TUNING ____________________________________________
    # cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=0)
    # space = dict()
    # space['hidden_layer_sizes'] = [(10, 3), (3, 10), (5, 5)]
    # space['activation'] = ['relu', 'tanh', 'logistic']
    # space['solver'] = ['bfgs', 'sgd', 'adam']
    # space['learning_rate_init'] = [0.001, 0.01, 0.0001]
    # search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
    # result = search.fit(X_train, y_train)
    # print('Best Score: %s' % result.best_score_)
    # print('Best Hyperparameters: %s' % result.best_params_)



    # _______________________________________________ PARAMETER TRAINING _______________________________________________
    # MLP
    MLP.fit(X_train,y_train)

    # Linear Regression
    LR.fit(X_train,y_train)

    # Random Forest Regressor
    RF.fit(X_train,y_train)

    # _______________________________________________ INFERENCE _______________________________________________
    # MLP
    y_mlp = MLP.predict(X_test)

    # Linear Regression
    y_lr = LR.predict(X_test)

    # Random Forest Regressor
    y_rf = RF.predict(X_test)

    # _______________________________________________ ERROR MEASURE _______________________________________________
    mae = []
    mae.append(metrics.mean_absolute_error(y_test, y_mlp))
    mae.append(metrics.mean_absolute_error(y_test, y_lr))
    mae.append(metrics.mean_absolute_error(y_test, y_rf))

    mse = []
    mse.append(metrics.mean_squared_error(y_test, y_mlp))
    mse.append(metrics.mean_squared_error(y_test, y_lr))
    mse.append(metrics.mean_squared_error(y_test, y_rf))

    r2 = []
    r2.append(metrics.r2_score(y_test,y_mlp))
    r2.append(metrics.r2_score(y_test,y_lr))
    r2.append(metrics.r2_score(y_test,y_rf))