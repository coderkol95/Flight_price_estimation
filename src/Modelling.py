from hyperopt.base import STATUS_OK
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import Preprocessing
"""

Modelling:

Different machine learning models were trained on the train data and tested on the test data.
The top three models' hyperparameteres were tuned to obtain the best performance.
AdaBoost performed the best.

Here RandomForest is being trained on the whole dataset for deployment.

"""

def optimize(params):

    """
    Objectives
    1. Initialize model with received params
    2. For the 5 splits, fit it on the train and score it on the test
    3. Calculate the mean MAPE across the splits and return it 
    
    """    
    model=RandomForestRegressor(**params)
    k=KFold()
    mapes=[]
    
    for idx in k.split(X=X,y=y):
        train_idx,test_idx=idx[0],idx[1]
        
        xtrain=X[train_idx]
        ytrain=y[train_idx]

        xtest=X[test_idx]
        ytest=y[test_idx]

        model.fit(xtrain,ytrain)
        preds=model.predict(xtest)
        
        fold_acc = mean_absolute_percentage_error(ytest, preds)
        mapes.append(fold_acc)
    
    return np.mean(mapes)
    
if __name__=='__main__':

    """
    Objectives
    1. Read the pickle files for the pipeline and encoder structure
    2. Read the entire dataset and transform it after fitting the pickles
    3. Fit the pipeline and the encoder for transforming input data in deployed model
    4. Define the searchable parameter space for Bayesian optimization
    5. Create the optimization function
    6. Instantiate trials and result for storing the outcome of each trial
    7. Create the best Random Forest with best `result` from HyperOpt
    8. Dump the fitted pipeline, encoder and model into pickle files
    
    
    """
    #1
    with open('./bin/features.pkl','rb') as f1:
        features=pickle.load(f1)

    with open('./bin/encoder.pkl','rb') as f2:
        encoder=pickle.load(f2)

    #2
    dataset = pd.read_csv(r'./data/dataset.csv')
    X, y = dataset.iloc[:,:-1], dataset.iloc[:,-1]

    #3
    feats=features.fit(X)
    X=feats.transform(X)
    code=encoder.fit(X)
    X=code.transform(X)
'''
    #4
    rf_space = {
            "max_depth": scope.int(hp.quniform('max_depth',1,50,1)),
            "n_estimators": scope.int(hp.quniform('n_estimators',10,500,1)),
            "max_features": hp.uniform('max_features',0.01,1)
                }
    #5
    def score_hyperparams(params):
        scor = optimize(params)
        return {'loss':scor, 'status':STATUS_OK}

    #6
    trials = Trials()

    result = fmin(
                fn=score_hyperparams,
                space=rf_space,
                max_evals=10,
                trials=trials,
                algo=tpe.suggest
            ) 
    print(result)

    #7

    #model = RandomForestRegressor(max_depth=int(result['max_depth']), max_features=result['max_features'], n_estimators=int(result['n_estimators']))
    model = RandomForestRegressor(max_depth=17, max_features=0.6807119650079912, n_estimators=192)

    model.fit(X,y)
'''
    with open(r'./bin/model.pkl','wb') as f3:
        pickle.dump(model,f3)

    #8
    with open(r'./bin/feats.pkl','wb') as f1:
        pickle.dump(feats,f1)

    with open(r'./bin/code.pkl','wb') as f2:
        pickle.dump(code,f2)
'''
