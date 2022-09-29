
import pandas as pd
d = {'x1': [1, 4, 4, 5, 6], 
     'x2': [0, 0, 8, 2, 4], 
     'x3': [2, 8, 8, 10, 12], 
     'x4': [-1, -4, -4, -4, -5]}
df = pd.DataFrame(data = d)
print("Data Frame")
print(df)
print()

print("Correlation Matrix")
print(df.corr())
print()

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 3))




        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],), name="Input-Layer"),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(X_test.shape[1], activation='softmax')
        ])
        # print(model.summary())
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                      loss='mean_absolute_error')
        history = model.fit(X_train,
                            y_train,
                            epochs=50,
                            verbose=0,
                            #validation_split=0.2
                            )
        score = model.evaluate(X_test, y_test, verbose=0)
        mse = mean_squared_error(model.predict(X_test), y_test)
        mae = mean_absolute_error(model.predict(X_test), y_test)
        return model, score, mse, mae
    
    
    
    #### Train a neural network #TODO:
    def neural_network_model(X_train, X_test, y_train, y_test):
        
        def create_model(init_mode='uniform'):
            
            model = Sequential() # create model
            
            model.add(Dense(12, 
                            input_shape=(X_train.shape[1],), 
                            # kernel_initializer=init_mode, 
                            activation='relu'))
            model.add(Dense(1, 
                            # kernel_initializer=init_mode, 
                            activation='sigmoid'))
            
            model.compile(loss='mean_squared_error', 
                          optimizer='adam', 
                          metrics=['accuracy'])
            
            return model
        
        
    def SVM_model(X_train, X_test, y_train, y_test):
        model = SVR(#kernel="poly",
                    #degree=3,
                    C=50,
                    epsilon=0.1).fit(X_train, y_train) # fit model with training data
        # FIXME: y should be a 1d array, got an array of shape (138, 3) instead.
        # TODO: kann glaube ich nur einen 1-D output lifern ....
        # evaluate model on training- and test-data
        score = model.score(X_train, y_train)  # R2-score
        mse = mean_squared_error(model.predict(X_test), y_test)
        mae = mean_absolute_error(model.predict(X_test), y_test)
        return model, score, mse, mae