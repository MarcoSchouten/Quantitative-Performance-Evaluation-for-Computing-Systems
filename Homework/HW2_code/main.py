import time


from scipy.stats import loguniform
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd  # used to load the data
import numpy as np  # optimized numerical library
from scipy import stats
from sklearn import preprocessing, metrics, utils, decomposition, model_selection, linear_model, discriminant_analysis, svm, tree, ensemble
# library providing several ML algorithms and related utility
from sklearn.neural_network import MLPClassifier
from imblearn import over_sampling  # provides several resampling techniques to cope with unbalanced datasets (https://github.com/scikit-learn-contrib/imbalanced-learn) compatible with sklearn
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import matplotlib.pyplot as plt


# Start by defining three helper functions:
# - one to plot the sample distribution  acorss the class labels (to see how un-/balanced the dataset is)
# - one to compute and plot the confusion matrix
# - one to plot data in 2D with different colors per class label

def plot_pie(y, labels, title=""):
    target_stats = Counter(y)
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))

    fig, ax = plt.subplots()
    ax.set_title(title + " (size: %d)" % len(y))
    ax.pie(sizes, explode=explode, labels=target_stats.keys(), shadow=True, autopct='%1.1f%%')
    ax.axis('equal')


def compute_and_plot_cm(ytest, ypred, labels, title=""):
    global nfigure
    # Compute confusion matrix
    cm = metrics.confusion_matrix(ytest, ypred)

    accuracy = metrics.accuracy_score(ytest, ypred, normalize=True)

    # Normalize the matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    # Plot the confusion matrix

    nfigure = nfigure + 1
    plt.figure(nfigure)  # new numbered figure
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # plot the confusionmatrix using blue shaded colors
    plt.title("Confusion Matrix Normalized (%s) Accuracy: %.1f%%" % (title, accuracy * 100))  # add title
    plt.colorbar()  # plot the color bar as legend

    # Plot the x and y ticks using the class label names
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)


def plot_2d(xpred, ypred, labels, title=""):
    global nfigure
    # define the colors to use for each class label
    colors = ['red', 'blue', 'green', 'yellow', 'black']
    len_colors = len(colors)
    if len_colors < len(labels):
        print("WARNING: we have less colors than classes: some classes will reuse the same color")

    nfigure = nfigure + 1
    plt.figure(nfigure)  # new numbered figure
    plt.title("Feature Space (%s)" % title)  # add title

    # plot each class label with a separate color
    for c in [4, 5]:
        cur_class = (ypred == c)  # get all points belonging to class c
        plt.plot(xpred[cur_class, 0], xpred[cur_class, 1], 'o', color=colors[c % len_colors])  # plot class c


nfigure = 0  # used to number the figures


 #_______________________________________________ (1) LOAD DATA _______________________________________________
# Get the dataset loaded and define class labels
data = pd.read_csv('jobs.csv', header=0)
data_class_labels = ["successful", "unsuccessful"]
# print(data.head())

# All data columns except last are input features (X), last column is output label (y)
n_features = len(data.columns) - 1

X = data.iloc[:,0:n_features].values
y = data.iloc[:,n_features].values
X = X[y != 2]
y = y[y != 2]
X = X[y != 3]
y = y[y != 3]

# plot_pie(y, data_class_labels, "Original")
# plt.show()

# _______________________________________________ (2) BALANCE DATA _______________________________________________
################ Resample data #################
# Google data is very skewed, try to balance the dataset
# sm = over_sampling.SMOTE(random_state=42, ratio="auto")
# X_balanced, y_balanced = sm.fit_sample(X,y)
# ros = RandomOverSampler(random_state=42, sampling_strategy='auto')
# X_res, y_res = ros.fit_resample(X, y)
#
# # Plot the balanced label distribution
# plot_pie(y,data_class_labels, "Balanced")
# plt.show()
ros = RandomOverSampler(random_state=42, sampling_strategy=0.99)
X_res, y_res = ros.fit_resample(X, y)
# plot_pie(y_res,data_class_labels, "Balanced")
# plt.show()

# _______________________________________________ (3) SMALLER SUBSET _______________________________________________
# Resample the data with simple random resampling (if too big)
# - replace decideds if sampling with or without replacement
# - n_samples decide the size of the ouput: if set to None ouput = input (i.e. no resampling)
X_small, y_small = utils.resample(X_res,y_res, replace=False, n_samples=5000)
# Plot the resampled label distribution
# plot_pie(y_small,data_class_labels, "Sampled")
# plt.show()






# _______________________________________________  remove outliers _______________________________________________
# data_SMALL_dict = {'c1':  X_small[:,0],
#         'c2':  X_small[:,1],
#         'c3':  X_small[:,2],
#         'c4': X_small[:, 3],
#         'c5': X_small[:, 4],
#         'c6': X_small[:, 5],
#         'c7': X_small[:, 6],
#         'c8': X_small[:, 7],
#         'c9': X_small[:, 8],
#         'c10': X_small[:, 9],
#         'c11': X_small[:, 10],
#         'c12': X_small[:, 11],
#         'c13': X_small[:, 12],
#         'c14': X_small[:, 13],
#         'c15': X_small[:, 14],
#         'c16': X_small[:, 15],
#         'c17': X_small[:, 16],
#         'c18': X_small[:, 17],
#         'c19': X_small[:, 18],
#         'c20': X_small[:, 19],
#         'c21': X_small[:, 20],
#         'c22': X_small[:, 21],
#         'c23': X_small[:, 22],
#         'c24': X_small[:, 23],
#         'c25': X_small[:, 24],
#         'c26': X_small[:, 25],
#         'c27': X_small[:, 26],
#         'c28': X_small[:, 27],
#         }
# df_small = pd.DataFrame(data_SMALL_dict)
# df_small['y'] = y_small
# print(df_small.head())
# print(df_small.shape)
#
# z_scores = stats.zscore(df_small)
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# new_df = df_small[filtered_entries]
# # print(new_df.head())
#
#
# X_o = data.iloc[:,0:n_features].values
# y_o = data.iloc[:,n_features].values
# plot_pie(y_o,data_class_labels, "outliers")
# plt.show()



# _______________________________________________ (4)SCALE _______________________________________________
################ Scale data ####################
# Train a scaler to standardize the features (zero mean and unit variance)
X_small = preprocessing.StandardScaler().fit_transform(X_small)
# print(np.mean(X_small),np.std(X_small))



# _______________________________________________ (4) DIMENSIONALITY  REDUCTION ______________________
# Train a PCA with k dimensions: Write a script to find minimum number of k components that ...
# contain 90% of variance description

feat_cols = ['feature'+str(i) for i in range(X_small.shape[1])]
small_df = pd.DataFrame(X_small,columns=feat_cols)
# print(small_df.head())

n_components=16
pca_small = decomposition.PCA(n_components)
principalComponents_small = pca_small.fit_transform(X_small)
pca_cols = ['pc'+str(i) for i in range(n_components)]
principal_small_Df = pd.DataFrame(data = principalComponents_small, columns = pca_cols)
# print(principal_small_Df.tail())
# print('Explained variation per principal component: {}'.format(pca_small.explained_variance_ratio_))
sum_Var = np.sum(pca_small.explained_variance_ratio_) * 100
print('SumVar'+str(np.round(sum_Var,0)))
X_small = principal_small_Df.iloc[:,0:n_components]


# _______________________________________________ (4) SPLIT DATA _______________________________________________
# Split data in training and testing for 0.33 ratio of testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_small, y_small, test_size=0.33, random_state=42)
print('X train {}'.format(X_train.shape) )
print('X test{}'.format(X_test.shape))
print('y train {}'.format(y_train.shape))
print('y train {}'.format(y_test.shape))


# _______________________________________________ (5) MODEL HYPERPARAMETER SEARCH ____________________________
# LOGISTIC REGRESSION
model =  linear_model.LogisticRegression()
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)
# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
# execute search
result = search.fit(X_small, y_small)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)



# SUPPORT VECTOR MACHINE
model = svm.SVC()
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
space = dict()
space['gamma'] = [1e-2, 1e-1, 1, 10, 100]
space['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
space['C'] = [1e-2, 1e-1, 1, 10, 100]
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
result = search.fit(X_train, y_train)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)



# DECISION TREE
model = tree.DecisionTreeClassifier()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
space = dict()
space['criterion'] = ['gini','entropy']
space['splitter'] = ['best', 'random']
space['max_depth'] = [2,21,32]
space['min_samples_split'] = [0.1, 0.2, 0.4, 0.6]
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
result = search.fit(X_train, y_train)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)



# RANDOM FOREST
model =  ensemble.RandomForestClassifier()
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
space = dict()
space['bootstrap'] = [ True, False ]
space['max_depth'] = [2,21,32]
space['max_features'] = [ 8, 12, 16 ]
space['min_samples_leaf'] = [0.01, .1, 0.2, ]
space['min_samples_split'] = [0.1, 0.4, 0.6]
space['n_estimators'] = [ 10, 50 , 100 ]

search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
result = search.fit(X_train, y_train)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# feature_importance
rf =  ensemble.RandomForestClassifier( n_estimators= 100, bootstrap= False, max_depth= 21, max_features = 16,
                                       min_samples_leaf= 0.01, min_samples_split =0.1 )
rf.fit(X_train, y_train)
plt.barh(pca_cols, rf.feature_importances_)
print(rf.feature_importances_)
plt.show()




#  MLP CLASSIFIER

# hyperparam tuning
model = MLPClassifier()
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=0)
space = dict()
space['hidden_layer_sizes'] = [ (10, 3)  , (3,10),  (5,5) ]
space['activation'] = ['relu' , 'tanh' ,'logistic']
space['solver'] = ['bfgs', 'sgd', 'adam']
space['learning_rate_init'] = [0.001, 0.01, 0.0001]
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
result = search.fit(X_train, y_train)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
