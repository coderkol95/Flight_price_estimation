import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from datetime import datetime as dt
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
import pickle


class date_splitter(BaseEstimator,TransformerMixin):
    """
    Objectives:
    1. Convert string to datetime
    2. Extract the date of journey
    3. Extract the day of week
    4. Extract the month of the year
    5. Extract the day of the year
    6. Drop the composite date and return these new components
    
    """
    def __init__(self,Date_of_Journey):
        self.Date_of_Journey=Date_of_Journey

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):

        X['Date_of_Journey'] = X['Date_of_Journey'].apply(lambda x: dt.strptime(str(x),'%m/%d/%y'))
        X['Day of month'] = X['Date_of_Journey'].apply(lambda x: x.strftime("%d")).astype(int)
        X['Day of week'] = X['Date_of_Journey'].apply(lambda x: x.strftime("%w")).astype(int)
        X['Month of year'] = X['Date_of_Journey'].apply(lambda x: x.strftime("%m")).astype(int)
        X['Day of year'] = X['Date_of_Journey'].apply(lambda x: x.timetuple().tm_yday)

        X.drop(['Date_of_Journey'],axis=1,inplace=True)
        return X[['Day of month','Day of week','Month of year','Day of year']].values

class route(BaseEstimator,TransformerMixin):
    
    """
    Objectives:
    1. Generate unique route based on the source and destination
    2. Drop the source and destination and return the route    
        
    """    
   
    def __init__(self,Source,Destination):
        self.Source=Source
        self.Destination=Destination

    def fit(self,X,y=None):
        return self

    def transform(self, X, y=None):

        mapper={
        'BangaloreNew Delhi':1, 
        'ChennaiKolkata':2, 
        'New DelhiCochin':3,
        'KolkataBangalore':4, 
        'MumbaiHyderabad':5}        

        X['route'] = X['Source']+X['Destination']

        X['route'] = X['route'].map(mapper)
        X['route'] = X['route'].apply(lambda x:int(x))

        X.drop(['Source','Destination','Route','Additional_Info','Arrival_Time'],axis=1,inplace=True)
        
        return X[['route']].values

class time_trier(BaseEstimator,TransformerMixin):
    
    """
    Objectives:
    1. Extract the no. of hours the flight flew
    2. Extract the minutes of flight duration
    3. Return these as features after dropping the initial feature
    
    """
    
    
    def __init__(self, Duration):
        self.Duration= Duration

    def fit(self,X,y=None):
        return self

    def transform(self, X, y=None):
             
        dur_hour = lambda x:x[:x.index("h")] if 'h' in x else 0
        dur_min = lambda x: x[x.index("m")-2:x.index("m")] if 'm' in x else 0

        X['Duration_hours'] = X.Duration.apply(dur_hour)
        X['Duration_mins'] = X.Duration.apply(dur_min)
        
        X.Duration_mins.replace({'':'0'},inplace=True)
          
        X['Duration_hours'] = X['Duration_hours'].apply(lambda x:int(x))
        X['Duration_mins'] = X['Duration_mins'].apply(lambda x:int(x))

        X.drop(['Duration'],axis=1,inplace=True)
        
        return X[['Duration_hours','Duration_mins']].values

class tod_departure(BaseEstimator,TransformerMixin):
    
    """
    Objectives:
    1. Extract the departure hour
    2. Extract the departure minutes
    3. Depending on the departure hour, generate a new feature TOD='Time of day'
    4. Return these three new features and drop the original feature
    """
    def __init__(self, Dep_Time):
        self.Dep_Time = Dep_Time

    def fit(self,X,y=None):
        return self

    def transform(self, X, y=None):
        
        hour = lambda x: x[:x.index(":")]        
        minutes = lambda x: x[x.index(":")+1:]

        X['Dep_hour'] = X.Dep_Time.apply(hour)
        X['Dep_minutes'] = X.Dep_Time.apply(minutes)

        X['Dep_minutes'] = X['Dep_minutes'].apply(lambda x: int(x))
        X['Dep_hour'] = X['Dep_hour'].apply(lambda x:int(x))
     
        tod = lambda x: 'early morning' if 0<x<=6 else('morning' if 6<x<=12 else ('noon' if 12<x<=16 else ('evening' if 16<x<=20 else 'night')))
        X['TOD'] = X.Dep_hour.map(tod)
        X.drop(['Dep_Time'],axis=1,inplace=True)
        
        return X[['TOD','Dep_minutes','Dep_hour','Airline']].values

class filters(BaseEstimator,TransformerMixin):
    
    """
    Objectives:
    
    Filter the non-stop flights.

    """
    def __init__(self, Total_Stops):
        self.Total_Stops=Total_Stops

    def fit(self,X,y=None):
        return self

    def transform(self, X, y=None):

        non_stop={'non-stop':1, np.nan:1, '2 stops':0, '1 stop':0, '3 stops':0,'4 stops':0}
        X.Total_Stops = X.Total_Stops.map(non_stop)
        
        X = X[X.Total_Stops==1]

        return X.values

"""
Creating the custom pipeline.

"""
features=FeatureUnion(
    transformer_list=[
        ('date_spliiter',date_splitter('Date_of_Journey')),
        ('route_identifier', route('Source','Destination')),
        ('timer', time_trier('Duration')),
        ('time of departure',tod_departure('Dep_Time')) ])


"""
After new features are generated, this transformer is used to encode two features. 
This could not be fitted into the pipeline due to two reasons
1. The no. of features are expanding during the pipeline
2. TOD is generated. Encoding cannot be used before it

"""
encoder=ColumnTransformer([('airline_TOD',OneHotEncoder(
    ),
    [10,7])], remainder='passthrough')


#Filtering only direct flights
pipe=Pipeline([('filter_hopping_flights', filters('Total_Stops'))])

#Reading the entire dataset and filtering the non-stop flights
df=pd.read_csv(r'./data/flight_price.csv')
dataset=pd.DataFrame(pipe.fit_transform(df))
dataset.columns = df.columns

y=dataset.iloc[:,-1]
X=dataset.iloc[:,:-1]

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
trainset = pd.concat([X_train,y_train],axis=1)
testset = pd.concat([X_test,y_test],axis=1)

#The column names are necessary for the custom transformer
trainset.columns = df.columns
testset.columns = df.columns
y_train = pd.DataFrame(trainset.Price)
X_train = trainset.drop(['Price'], axis=1)
y_test = pd.DataFrame(testset.Price)
X_test = testset.drop(['Price'], axis=1)


#Generating the features using custom pipeline
features.fit(X)
X = pd.DataFrame(features.transform(X))
X_test = pd.DataFrame(features.transform(X_test))

#Encoding the features
encoder.fit(X)
X = pd.DataFrame(encoder.transform(X))
X_test = pd.DataFrame(encoder.transform(X_test))

#Concatenating them to write them to files for model assessment
trainset = pd.concat([X,pd.DataFrame(y_train.values)],axis=1)    
testset  = pd.concat([X_test, pd.DataFrame(y_test.values)],axis=1)

#Outlier removal using Z score method
trainset['std_price'] = (trainset.iloc[:,-1]-trainset.iloc[:,-1].mean())/trainset.iloc[:,-1].std()
trainset = trainset[(trainset['std_price']<3) & (trainset['std_price']>-3)]
trainset.drop(['std_price'],axis=1,inplace=True)
 
#Writing the transformed data out
dataset.to_csv(r'./data/dataset.csv',index=False)
trainset.to_csv(r'./data/trainset.csv',index=False)
testset.to_csv(r'./data/testset.csv',index=False)

#Dumping the feature generation into a pickle for fitting the entire dataset later
with open(r'./bin/features.pkl','wb') as f1:
    pickle.dump(features,f1)

#Dumping the encoding transformers into a pickle for fitting the entire dataset later
with open(r'./bin/encoder.pkl','wb') as f2:
    pickle.dump(encoder,f2)