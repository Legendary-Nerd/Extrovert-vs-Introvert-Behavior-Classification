import streamlit as st
import numpy as np
import joblib


st.title("Project AiMl")
st.header("Introvert vs Extrovert")


time_spent_alone = st.slider("Time Spent Alone :", 0,0,11)
stage_fear = st.selectbox("Stage Fear Level :", ["Yes", "No"])
if stage_fear == "Yes":
    stage_fear = 1
else:
    stage_fear = 0
social_event_attendance = st.slider("Social Events per Month", 0 ,0,10)
going_outside = st.slider("Go Outside per Week", 0 ,0,7)
drained_after_socializing = st.selectbox("Tired After Socializing", ["Yes", "No"])
if drained_after_socializing == "Yes":
    drained_after_socializing = 1
else:
    drained_after_socializing = 0
friends_circle_size = st.number_input("Number of Close Friends", value=0.0)
post_frequency = st.slider("social post", 0,0,10)

features = np.array([time_spent_alone,stage_fear,social_event_attendance,going_outside,drained_after_socializing,friends_circle_size,post_frequency]).reshape(1, -1)


def Y_predict(model_name, features):
    avg = np.mean(features)
    if avg > 5:
        return f"{model_name} predicts: You are likely an Extrovert"
    else:
        return f"{model_name} predicts: You are likely an Introvert"


model_list = ["SVM", "Decision Tree", "Linear Regression", "Perceptron"]

# Create spacing columns to center buttons
spacer, col1, col3, col4, spacer2 = st.columns([1, 2, 2, 2, 1])
iris_classes = ['Extrovert','Introvert']
# Put one button in each column
with col1:
    if st.button("SVM"):
        model = joblib.load('./models/svm_pipeline.pkl')     
        prediction = model.predict(features)
        # answer = prediction
        
        st.subheader(f"SVM Prediction: {iris_classes[prediction[0]]}")

# with col2:
#     if st.button("Decision Tree"):
#         model = joblib.load('./models/decision_tree_pipeline.pkl')
#         prediction = model.predict(features)
#         st.success(f"Decision Tree Prediction: {iris_classes[prediction[0]]}")

with col3:
    if st.button("Linear Regression"):
        model = joblib.load('./models/linear_pipeline.pkl')
        prediction = model.predict(features)
        st.success(f"Linear Regression Prediction: {iris_classes[prediction[0]]}")

with col4:
    if st.button("Perceptron"):
        model = joblib.load('./models/mlp_pipeline.pkl')
        prediction = model.predict(features)
        st.success(f"Perceptron Prediction: , {iris_classes[prediction[0]]}")