import streamlit as st
import numpy as np
import tensorflow as tf 
import pandas as pd
import pickle

# load trained model
model = tf.keras.models.load_model('Model.h5')

# load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    onehot_geo_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# streamlit app
st.title('Customer churn prediction')

# user input
geography = st.selectbox('Geography', onehot_geo_encoder.categories_[0])
gender = st.selectbox('Gender', gender_encoder.classes_)
age = st.slider('Age',18,100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Nummber of Products',1,4)
has_cr_card = st.slectbox('Has Credit Card',[0,1])
is_active_member = st.slectbox('Is Active Member',[0,1])

# prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [gender],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# preprocessing 

geo_encoded = onehot_geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geo_encoder.get_feature_names_out(['Geography']))
# combining encoded col to input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale data
input_data_scaled = scaler.transform(input_data)

# predict chrun
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]


st.write(f'Chrun Probablity: {prediction_prob}')


if prediction >0.5:
    st.write('The Customer is likely to leave bank')
else:
    st.write('The Customer is not likely to leave bank')
