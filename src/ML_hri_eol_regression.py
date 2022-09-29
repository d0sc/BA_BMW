### TO SPECIFY ###
BaseDataSetPath = "data/prep/max_filtered_final.csv" # path for the raw dataset, which is used for training

# decide on which HRI-features to take as INPUT
x_features = ["hri_O_9.9", 
              "hri_O_13.31", 
              "hri_O_19.91", 
              "hri_O_20.02", 
              "hri_O_21.3", 
              "hri_O_47.18",  
              "hri_O_78.27", 
              "hri_O_78.38", 
              "hri_O_86.15", 
              "hri_O_86.26"]
# decide on which EOL-features to take as OUTPUT
y_features = ["eol_O_20.0"] # FINAL 1D


# imports 
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True, style="whitegrid")
from datetime import date
today = date.today()



def visualize_linear_regression(df, x_features, y_features):

    count_plots_horizontal = len(x_features)
    count_plots_vertikal = len(y_features)

    fig, axes = plt.subplots(count_plots_vertikal, count_plots_horizontal, figsize=(18, 8))
    fig.suptitle('Correlation-plots for all HRI-Ordnungen to all EOL-Ordnungen')
    fig.subplots_adjust(hspace=.5, wspace=.001)

    for row in range(count_plots_horizontal):
        for col in range(count_plots_vertikal):

            plot = sns.regplot(ax=axes[col, row],
                        x=df.loc[:, x_features[row]],
                        y=df.loc[:, y_features[col]],
                        data=df)
            plot.set(xticklabels=[], yticklabels=[])
            
            if col != 3:
                plot.set(xlabel=None)
            if row != 0:
                plot.set(ylabel=None)

    plt.show()



def get_hri_eol_model(df:pd.DataFrame, show_stuff:bool=False):

    # X-shape: [[-,-,-,-,-],
    #           [-,-,-,-,-]]
    # y-shape: [[-],
    #           [-]]
    
    
    #### Split the data into Train and test set
    def test_train_split(df):

        X = np.array(df.loc[:, x_features]).reshape(-1, len(x_features))    # (2859, 8)
        y = np.array(df.loc[:, y_features]).reshape(-1, len(y_features))    # (2859, 4)

        # split into train, test, val sets
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=0.2, 
                                                            random_state=1)  # FIXME: # no validation set, not enough samples yes

        if show_stuff:
            print("X-shape: ", X.shape)             # (173, 11)
            print("y-shape: ", y.shape)             # (173, 4)
            print("X_train-shape: ",X_train.shape)  # X_train.shape: (138, 3)
            print("X_test-shape: ",X_test.shape)    # X_test.shape: (35, 3)
            print("y_train-shape: ",y_train.shape)  # y_train.shape: (138, 3)
            print("y_test-shape: ",y_test.shape)    # y_test.shape: (35, 3)

        return X_train, X_test, y_train, y_test

    #### Train a linear regression model
    def multi_LR(X_train, X_test, y_train, y_test):
                
        model = LinearRegression().fit(X_train, y_train) # fit model with training data
        
        # evaluate model on training- and test-data
        score = model.score(X_train, y_train)  # R2-score
        mse = mean_squared_error(model.predict(X_test), y_test)
        mae = mean_absolute_error(model.predict(X_test), y_test)
        
        # evaluation["LR"][f"{iteration+1}"] = {"score": score,
        #                                    "mse": mse,
        #                                    "mae": mae}
        
        dump(model, f'model/LR/LR_model_{today}_{score.round(2)}_{mse.round(2)}_{mae.round(2)}.joblib') 
        
        print('LR - Training beendet')
        
        return model, score, mse, mae

    #### Train a decision tree model
    def DT_regresion(X_train, X_test, y_train, y_test):
        """
        https://www.section.io/engineering-education/hyperparmeter-tuning/
        """
        
        # std_slc = StandardScaler()  #FIXME: USE IT?

        dec_tree = DecisionTreeRegressor() # init the decision tree Regressor
        
        # create a pipeline for all the three objects std_scl, pca and dec_tree.
        pipe = Pipeline(steps=[('dec_tree', dec_tree)])
                            #    ('std_slc', std_slc),
        
        # DecisionTreeClassifier requires two parameters 'criterion' and 'max_depth' to be optimised by GridSearchCV
        criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        max_depth = [2,4,6,8,10,12]
        # splitter = ['best', 'random']
        min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
        min_samples_leaf = [1, 2, 4] # Minimum number of samples required at each leaf node
        max_features = ['auto', 'sqrt'] # Number of features to consider at every split

        parameters = dict(criterion=criterion,
                          max_depth=max_depth,
                        #   splitter=splitter,
                          min_samples_split=min_samples_split,
                          min_samples_leaf=min_samples_leaf,
                          max_features=max_features
                          )

        """
        Before using GridSearchCV, lets have a look on the important parameters.

        estimator:   In this we have to pass the models or functions on which we want to use GridSearchCV
        param_grid:  Dictionary or list of parameters of models or function in which GridSearchCV have to select the best.
        Scoring:     It is used as a evaluating metric for the model performance to decide the best hyperparameters, if not especified then it uses estimator score.
        """
        
        # clf_GS = GridSearchCV(pipe, parameters)
        clf_GS = RandomizedSearchCV(estimator = dec_tree, 
                                    param_distributions = parameters, 
                                    n_iter = 200, 
                                    cv = 3, 
                                    verbose=0, 
                                    random_state=42, 
                                    n_jobs = -1)


        clf_GS.fit(X_train, y_train)
        
        print('DT - Best Criterion:', clf_GS.best_estimator_.get_params()['criterion'])
        print('DT - Best max_depth:', clf_GS.best_estimator_.get_params()['max_depth'])
        # print('DT - Best splitter:', clf_GS.best_estimator_.get_params()['splitter'])
        print('DT - Best min_samples_split:', clf_GS.best_estimator_.get_params()['min_samples_split'])
        print('DT - Best min_samples_leaf:', clf_GS.best_estimator_.get_params()['min_samples_leaf'])
        print('DT - Best max_features:', clf_GS.best_estimator_.get_params()['max_features'])
        
        score = clf_GS.score(X_train, y_train)  # R2-score
        mse = mean_squared_error(clf_GS.predict(X_test), y_test)
        mae = mean_absolute_error(clf_GS.predict(X_test), y_test)
        
        dump(clf_GS, f'model/DT/DT_model_{today}_{score.round(2)}_{mse.round(2)}_{mae.round(2)}.joblib')
        
        # evaluation["DT"] = {"score": score,
        #                     "mse": mse,
        #                     "mae": mae}

        return clf_GS, score, mse, mae
    
    ### Train Random forest model
    def RF_regresion(X_train, X_test, y_train, y_test):
        y_train = y_train.reshape(-1)
        """
        For detailed comments on the pipeline, check the decision tree regressor
        https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        """
        
        rand_for = RandomForestRegressor()
        
        pipe = Pipeline(steps=[('rand_for', rand_for)])
                            #    ('std_slc', std_slc),
                            #    ('pca', pca),
                            
        n_estimators = [100, 200, 400, 600, 800, 1000] # Number of trees in random forest
        max_features = ['auto', 'sqrt'] # Number of features to consider at every split
        max_depth = [10, 20, 40, 60, 80, 90, 100, 110] # Maximum number of levels in tree
        min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
        min_samples_leaf = [1, 2, 4] # Minimum number of samples required at each leaf node
        # bootstrap = [True, False] # Method of selecting samples for training each tree

        random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf}     

        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = rf,
                                       param_distributions = random_grid, 
                                       n_iter = 200,
                                       cv = 3, 
                                       verbose=0, 
                                       random_state=42, 
                                       n_jobs = -1)
        # rf_random = GridSearchCV(estimator = rf, 
        #                          param_grid= random_grid)
        #                         #  cv = 3, 
        #                         #  verbose=0, 
        #                         #  random_state=42, 
        #                         #  n_jobs = -1)
        
        rf_random.fit(X_train, y_train)
        
        score = rf_random.score(X_train, y_train)  # R2-score
        mse = mean_squared_error(rf_random.predict(X_test), y_test)
        mae = mean_absolute_error(rf_random.predict(X_test), y_test)
        
        dump(rf_random, f'model/RF/RF_model_{today}_{score.round(2)}_{mse.round(2)}_{mae.round(2)}.joblib')

        print('RF - Best n_estimators:', rf_random.best_params_['n_estimators'])
        print('RF - Best max_features:', rf_random.best_params_['max_features'])
        print('RF - Best max_depth:', rf_random.best_params_['max_depth'])
        print('RF - Best min_samples_split:', rf_random.best_params_['min_samples_split'])
        print('RF - Best min_samples_leaf:', rf_random.best_params_['min_samples_leaf'])
        
        return rf_random, score, mse, mae
       


    ##########################################################################
    ########################## EXECUTION CODE ################################
    ##########################################################################
    
    X_train, X_test, y_train, y_test = test_train_split(df)
    
    LR_model, LR_score, LR_mse, LR_mae = multi_LR(X_train, X_test, y_train, y_test)
    DT_model, DT_score, DT_mse, DT_mae = DT_regresion(X_train, X_test, y_train, y_test)
    RF_model, RF_score, RF_mse, RF_mae = RF_regresion(X_train, X_test, y_train, y_test)

    
    
    # Linear regression
    print("LR_model_score: ", LR_score)
    print("LR_model_mse: ", LR_mse)
    print("LR_model_mae: ", LR_mae)
    # decision tree
    print("DT_model_score: ", DT_score)
    print("DT_model_mse: ", DT_mse)
    print("DT_model_mae: ", DT_mae)
    # random forest
    print("RF_model_score: ", RF_score)
    print("RF_model_mse: ", RF_mse)
    print("RF_model_mae: ", RF_mae)



def make_prediction(x, model):

    prediction = model.predict(x)  #sklearn standart functionality

    return prediction




if __name__ == "__main__":
    
    df = pd.read_csv(BaseDataSetPath)
    # get_hri_eol_model(df)
    
    # print([int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)])

    LR_model = load("model/LR/LR_model_2022-09-22_0.79_7.24_1.89.joblib") # this is a LR model
    DT_model = load('model/DT/DT_model_2022-09-27_0.98_7.42_1.42.joblib') #DT
    RF_model = load('model/RF/RF_model_2022-09-27_0.97_5.15_1.25.joblib') #RF
    test_point = np.random.randint(low=10, high=100, size=10)
    predi = make_prediction(x=test_point.reshape(1, -1), model=DT_model)
    predi = make_prediction(x=test_point.reshape(1, -1), model=RF_model)
    
    
    #### Split the data into Train and test set
    def test_train_split(df):

        X = np.array(df.loc[:, x_features]).reshape(-1, len(x_features))    # (2859, 8)
        y = np.array(df.loc[:, y_features]).reshape(-1, len(y_features))    # (2859, 4)

        # split into train, test, val sets
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=0.2, 
                                                            random_state=1)  # FIXME: # no validation set, not enough samples yes

        return X_train, X_test, y_train, y_test
    
    X_train, X_test, y_train, y_test = test_train_split(df)
    
    lr_rmse = mean_squared_error(LR_model.predict(X_test), y_test, squared=False)
    dt_rmse = mean_squared_error(DT_model.predict(X_test), y_test, squared=False)
    rf_rmse = mean_squared_error(RF_model.predict(X_test), y_test, squared=False)
    
    print("LR rmse: ", lr_rmse)
    print("DT rmse: ", dt_rmse)
    print("RM rmse: ", rf_rmse)