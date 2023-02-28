import sklearn
import streamlit as st
import pandas as pd
import pickle

teams=['Gujarat Titans', 'Rajasthan Royals', 'Lucknow Super Giants',
'Punjab Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
'Kolkata Knight Riders', 'Sunrisers Hyderabad', 'Delhi Capitals',
'Chennai Super Kings']


cities=['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
'Visakhapatnam', 'Bengaluru', 'Chandigarh', 'Jaipur', 'Indore',
'Bangalore', 'Kanpur', 'Rajkot', 'Raipur', 'Ranchi', 'Cuttack',
'Dharamsala', 'Nagpur', 'Johannesburg', 'Durban', 'Centurion',
'Bloemfontein', 'Port Elizabeth', 'Kimberley', 'East London','Cape Town']


pipe = pickle.load(open("PySpark_LR.pkl","rb"))

st.title("IPL Match Winner Predictor")

col1 , col2 = st.columns(2)

with col1 :
    batting_team=st.selectbox('Select the batting team',sorted(teams))
    
teams.remove(batting_team)

with col2:
    bowling_team=st.selectbox("Select the bowling team",sorted(teams))


selected_city = st.selectbox("Select the host city",sorted(cities))
target = st.number_input("Target", min_value = 0, step = 1)


col3, col4, col5, col6=st.columns(4)

with col3:
    score = st.number_input("Score", min_value = 0, step = 1)
with col4:
    overs = st.number_input("Overs completed", min_value = 0, max_value = 19, step = 1)
with col5:
    balls_completed = st.number_input("Balls bowled", min_value = 0, max_value = 6, step = 1)
with col6:
    #if wickets > 10:
        wickets = st.number_input("Wickets fallen", min_value = 0, max_value = 10, step = 1)
overs1 =  overs
if balls_completed == 6:
    overs += 1
else:
    overs = overs + balls_completed/10

if st.button("Predict Probability"):
    runs_left = target - score
    balls_left= 120 - (overs1 * 6 + balls_completed)
    wickets= 10 - wickets
    if overs == 0 and balls_completed == 0:
        crr = 0
    else:
        crr= score/(overs)
    if balls_left != 0:
        rrr =(runs_left * 6)/balls_left
    else:
        rrr = 9999999

    input_df = pd.DataFrame({'batting_team' :[batting_team],
                             'bowling_team':[bowling_team],
                             'city':[selected_city],
                             'target_score': [target],
                             'runs_left':[runs_left],
                             'balls_left':[balls_left],
                             'wickets_left':[wickets],
                             'current_run_rate':[crr],
                             'required_run_rate':[rrr]})


    if target <= score:
        result = [[0.00, 1.00]]
    else:
        result = pipe.predict_proba(input_df)

    
    loss = result[0][0]
    win = result[0][1]
    if win > loss:
        st.header(batting_team + " - " + str(round(win * 100, 2)) + "% :sports_medal:")
        st.header(bowling_team + " - " + str(round(loss * 100, 2)) + "%")
    elif win < loss:
        st.header(batting_team + " - " + str(round(win * 100, 2)) + "%")
        st.header(bowling_team + " - " + str(round(loss * 100, 2)) + "% :sports_medal:")
    else:
        st.header(batting_team + " - " + str(round(win * 100, 2)) + "% :thinking_face:")
        st.header(bowling_team + " - " + str(round(loss * 100, 2)) + "% :thinking_face:")
 
