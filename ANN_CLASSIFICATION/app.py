import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
from tensorflow.keras.models import load_model
import streamlit as st
import tensorflow as tf

# load the train model , scalar pickle, onehot pickle
model=load_model('model.h5')

#load the encoders and scalar
with open('LabelEncoder_gender.pkl','rb') as file:
    lb=pickle.load(file)


with open('OneHotEncoder_geo.pkl','rb') as file:
    oh = pickle.load(file)

with open('scalar.pkl','rb') as file:
    scalar= pickle.load(file)

## streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', oh.categories_[0])
gender = st.selectbox('Gender', lb.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [lb.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]

})

geo_encoded = oh.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=oh.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scale = scalar.transform(input_data)

#prediction
prediction=model.predict(input_scale)
prediction_proba = prediction[0][0]


st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')