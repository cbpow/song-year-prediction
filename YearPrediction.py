import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Read the CSV file
df = pd.read_csv('YearPredictionMSD.txt', header=None)
X = df.drop(0, axis=1)

#Split into training and test sets
X_train = X[0:463715]
X_test = X[463715:]
y_train = df.iloc[:463715][0]
y_test = df.iloc[463715:][0]


#PCA
#Standardize the data before PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Apply PCA
pca = PCA(n_components=65)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)  

# #Plot of PCA components vs explained variance 
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# #Plot
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
# plt.axhline(y=0.95, color='r', linestyle='--')  # 95% variance line
# plt.xlabel("Number of Principal Components")
# plt.ylabel("Cumulative Explained Variance")
# plt.title("Choosing the Number of PCA Components")
# plt.grid()
# plt.show()


# #Searching for the best parameters for xgboost
# param_dist = {
#     "n_estimators": [500], #500
#     "max_depth": [7], #7
#     "learning_rate": [0.05], #0.05
#     "subsample": [0.6, 0.8, 1.0], #
#     "colsample_bytree": [0.8], #0.8
#     "reg_alpha": [0, 0.01, 0.1],
#     "reg_lambda": [1, 1.5, 2.0]
# }

# xgb = XGBRegressor(random_state=42)
# search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=50, scoring='r2', cv=3, verbose=2, n_jobs=-1)
# search.fit(X_train, y_train)
# best_model = search.best_estimator_
# print(best_model)


# Initialize classifiers
reg = LinearRegression()
enet = ElasticNet(random_state=0)
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
rf = RandomForestRegressor(n_estimators=100, random_state=40, min_samples_split=2, min_samples_leaf=4)
knn = KNeighborsRegressor(n_neighbors=5)
nn = MLPRegressor(random_state=5, max_iter=2000, tol=0.1)
xgb = XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bytree=0.8, random_state=42)


#Train and evaluate classifiers
classifiers = {
    'Linear Regression': (reg, X_train, X_test),
    'Elastic Net': (enet, X_train, X_test),
    'Ridge Regression': (ridge, X_train, X_test),
    'Lasso Regression': (lasso, X_train, X_test),
    # 'Random Forest': (rf, X_train_pca, X_test_pca), #commented out because of runtime
    # 'KNN': (knn, X_train_pca, X_test_pca),
    # 'Neural Network': (nn, X_train_pca, X_test_pca),
    'XGBoost': (xgb, X_train, X_test), 
}

for name, (clf, X_train_data, X_test_data) in classifiers.items():
    prev_time = time.time()
    print(f"Training {name}...")
    clf.fit(X_train_data, y_train)
    y_pred = clf.predict(X_test_data)

    training_time = time.time() - prev_time

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    #Results
    print(f"MAE: {mae:.2f} years")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f} years")
    print(f"R² Score: {r2:.4f}")
    print(f"Time taken: {training_time:.2f} seconds")
    print("-" * 50)
