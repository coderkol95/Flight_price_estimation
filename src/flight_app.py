import datetime
import streamlit as st
import pandas as pd
import pickle
import Preprocessing
#import Flight_price.src.Modelling


#Reading the duration for different routes
h=pd.read_csv(r"./data/hour_calculation.csv")


#Console I/O
st.title("Flight price prediction module for the awesome touring company")

#Date
dat=st.date_input("When do you plan to take the flight?",datetime.date(2020, 5, 17))
d=dat
dat=str(dat.strftime('%m/%d/%y'))

#Time
tim = st.time_input(f"At what time are you planning to fly on {d.strftime('%d/%m/%y')}?",datetime.time(7,1))
tim = str(tim.strftime('%H:%M'))

#Airline
airline = st.selectbox('Preferred choice of airline',('IndiGo', 'Air India', 'SpiceJet','GoAir', 'Vistara', 'Air Asia'))

#Getting the source based on the chosen airline
sourcelist=h[h['Airline']==airline].Source.values.tolist()
src = st.selectbox('Source',sourcelist)

#Getting the destination based on the chosen airline and source
destinationlist=h[(h['Airline']==airline) & (h['Source']==src)].Destination.values.tolist()
dest= st.selectbox('Destination',destinationlist)

st.write(f'Looking for flights from {src} to {dest} via {airline} on {dat} at {tim}... ')




def duration(src,dest,airline):
    
    """
    Returns the duration of the flight based on the source,destination and airline
    
    """
    identified = h[(h['Source']==src) & (h['Destination']==dest) & (h['Airline']==airline)]
    dur=identified.Duration.values[0]
    return dur

def processor(dat, tim, airline,src,dest):
    
    """
    Objectives:
    1.Fill dummy data for unnecessary fields
    2. Return the console data as a dataframe
    
    """
    airline=airline
    doj=dat
    src=src
    dest=dest
    route='asdf'
    dep_time=tim
    arr_time='10:20'

    
    dur=duration(src,dest,airline)
    addi='non-stop'
    stops=1

    data=[airline,doj,src,dest,route,dep_time,arr_time,dur,addi,stops]
    vase=pd.DataFrame(pd.Series(data,index=['Airline','Date_of_Journey','Source', 'Destination','Route', 'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops', 'Additional_Info'])).transpose()
    return vase

#Feature generation pickle to be loaded
with open('./bin/feats.pkl','rb') as f1:
    feats=pickle.load(f1)

#Encoding pickle to be loaded
with open('./bin/code.pkl','rb') as f2:
    code=pickle.load(f2)

#Model to be loaded
with open('./bin/model.pkl','rb') as f3:
    model=pickle.load(f3)

######################################################
#1. Get the entered data as a dataframe
#2. Generate features
#3. Do encoding
#4. Predict the output!
######################################################


input=processor(dat, tim, airline,src,dest)
input=feats.transform(input)
input=code.transform(input)
y=model.predict(input)[0]

#Get the duration of the flight
time=duration(src,dest,airline)


#Output the data to the console
st.text(f"The estimated price for the flight is between Rs. {round(1.11*y)} and {round(0.89*y)}")
st.text(f"Flight duration: {time}")
st.text("The prices won't be accurate as the model was trained on pre-Covid data.")