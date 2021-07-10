
# Business problem

Travel agencies need to track and predict fluctuations in flight prices to provide competitive tour packages. In this project we try to build a model to estimate flight prices for direct flights in five routes. This can be expanded to include more routes after more data is scraped from travel websites.


# Data preprocessing

## Available features

`Airline` : The flight carrier

`Date_of_Journey`: The date the flight flies

`Source`: Travel origin

`Destination`: Travel destination

`Route`: Whether the flight was hopping (Not used in modelling)

`Dep_Time`: Departure time at origin

`Arrival_Time`: Arrival time at destination (Not used in modelling)

`Duration`: Flight duration

`Total_Stops`: No. of stops (Not used in modelling)

`Additional_Info`: Miscellaneous information (Not used in modelling)

`Price`: Target continuous variable

## Data wrangling:

Custom transformers built for pipeline:

`date_splitter`: Extracts the day of week, day of month, month of year and day of year.

`route` : Maps the source and destination combinations into specific routes

`time_trier`: Extracts the hour and minutes from the flight duration data

`tod_departure`: Extracts the hour and minute details and generates a new feature `time of departure`

`filters`: Filters out hopping flights

These transformers are built into a pipeline. The transformer is then fitted on the train data and the transformed train and test sets are written to the disk. These are used in the modelling phase. There are two pipelines for feature engineering and encoding. These are written to pickle files for use in fitting the final dataset.

Outlier removal is done using Z score method.



# Model

## Model preparation and evaluation

I used MAPE as the evaluation metric. This was done so that the travel company's stakeholders could easily interpret the result and add this error in their travel package cost calculation. MAE or RMSE was not included because the flight prices were very different in different routes.

Based on training and test sets, three models were chosen for hyperparameter tuning:

|Sr. no.|Model|MAPE|
|--|--|--|
|1|Random Forest|9.98|
|2|Decision Tree|11.46|
|3|GBM|12.20|

After trying out different models and doing hyper parameter tuning, Random Forest performed the best. The fitted pipelines and the model are written out to pickle files in the disk for use in model deployment phase.

# Model deployment

The model was deployed using Streamlit. The following details were captured:
* Date of planned flight
* Time of planned flight
* Source
* Destination 
* Prefered choice of airline

Depending on the source and airline, the available destinations were displayed. The input data was first transformed through the pipelines. Then the model was used to predict the price and the expected flight duration.

Here is the app! : https://share.streamlit.io/coderkol95/flight_price/src/flight_app.py
