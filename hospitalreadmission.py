import streamlit as st
 
st.title('Hospital Readmission Prediction')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score
#from sklearn.preprocessing import LabelEncoder
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import GridSearchCV

model = joblib.load('Documents\model.pkl')

#file = "Downloads\hospital\hospital_readmissions.csv"

#df = pd.read_csv(file)

#st.write("The dataset: ")
#st.write(df)

#age = st.selectbox("Age", ["40-50", "50-60", "60-70", "70-80", "80-90", "90-100"])
st.write("Ages:")
st.write("40-50: 0, 50-60: 1, 60-70: 2, 70-80: 3, 80-90: 4, 90-100: 5")
age = st.number_input("Age", min_value=0, max_value=5, step=1)

time_hospital = st.number_input("Time in Hospital (Days)", min_value=0, step=1)
n_procedures = st.number_input("Number of Procedures", min_value=0, step=1)
n_labprocedures = st.number_input("Number of Lab Procedures", min_value=0, step=1)
n_medications = st.number_input("Number of Medications in Hospital", min_value=0, step=1)
n_outpatient = st.number_input("Number of Outpatient Visits before Hospital Stay", min_value=0, step=1)
n_inpatient = st.number_input("Number of Inpatient Visits before Hospital Stay", min_value=0, step=1)
n_emergency = st.number_input("Number of Emergency Visits before Hospital Stay", min_value=0, step=1)

st.write("Specialty of the Admitting Physician:")
st.write("Cardiology: 0, Emergency/Trauma: 1, Family/General Practice: 2, Internal Medicine: 3, Unsure: 4, Other: 5, Surgery: 6")
medical_specialty = st.number_input("Specialty of the Admitting Physician", min_value=0, max_value=6, step=1)
#medical_speciality = st.selectbox("Specialty of the Admitting Physician", ["Unsure", "Internal Medicine", "Family/General Practice", "Cardiology", "Surgery", "Emergency/Trauma", "Other"])

st.write("Primary Diagnosis:")
st.write("Circulatory: 0, Diabetes: 1, Digestive: 2, Injury: 3, Unsure: 4, Musculoskeletal: 5, Other: 6, Respiratory: 7")
diag_1 = st.number_input("Primary Diagnosis", min_value=0, max_value=7, step=1)
#diag_1 = st.selectbox("Primary Diagnosis", ["Unsure", "Circulatory", "Injury", "Digestive", "Respiratory", "Diabetes", "Musculoskeletal", "Other"])

st.write("Secondary Diagnosis:")
st.write("Circulatory: 0, Diabetes: 1, Digestive: 2, Injury: 3, Unsure: 4, Musculoskeletal: 5, Other: 6, Respiratory: 7")
diag_2 = st.number_input("Secondary Diagnosis", min_value=0, max_value=7, step=1)
#diag_2 = st.selectbox("Secondary Diagnosis", ["Unsure", "Circulatory", "Injury", "Digestive", "Respiratory", "Diabetes", "Musculoskeletal", "Other"])

st.write("Third Diagnosis:")
st.write("Circulatory: 0, Diabetes: 1, Digestive: 2, Injury: 3, Unsure: 4, Musculoskeletal: 5, Other: 6, Respiratory: 7")
diag_3 = st.number_input("Third Diagnosis", min_value=0, max_value=7, step=1)
#diag_3 = st.selectbox("Third Diagnosis", ["Unsure", "Circulatory", "Injury", "Digestive", "Respiratory", "Diabetes", "Musculoskeletal", "Other"])

st.write("Did the glucose serum came out as high (> 200), normal, or not performed?")
st.write("High: 0, Not Performed: 1, Normal: 2")
glucose = st.number_input("Glucose", min_value=0, max_value=2, step=1)
#glucose = st.selectbox("Did the glucose serum came out as high (> 200), normal, or not performed?", ["High", "Normal", "Not Performed"])

st.write("Did the A1C test came out as high (> 7%), normal, or not performed?")
st.write("High: 0, Not Performed: 1, Normal: 2")
a1c_test = st.number_input("A1C test", min_value=0, max_value=2, step=1)
#a1c_test = st.selectbox("Did the A1C test came out as high (> 7%), normal, or not performed?", ["High", "Normal", "Not Performed"])

st.write("Was there was a change in the diabetes medication?")
st.write("No: 0, Yes: 1")
change = st.number_input("Change in Medication", min_value=0, max_value=1, step=1)
#change = st.selectbox("Was there was a change in the diabetes medication?", ["Yes", "No"])

st.write("Was a diabetes medication prescribed?")
st.write("No: 0, Yes: 1")
med = st.number_input("Medication Prescribed", min_value=0, max_value=1, step=1)
#med = st.selectbox("Was a diabetes medication prescribed?", ["Yes", "No"]) 

if st.button("Predict Readmission"):
    st.write("The patient will be:")
    data = {'age': age, 'medical_specialty': medical_specialty, 'diag_1': diag_1, 'diag_2': diag_2, 'diag_3': diag_3, 'glucose_test': glucose, 'A1Ctest': a1c_test, 'time_in_hospital': time_hospital, 'n_lab_procedures': n_labprocedures, 'n_procedures': n_procedures, 'n_medications': n_medications, 'n_outpatient': n_outpatient, 'n_inpatient': n_inpatient, 'n_emergency': n_emergency, 'change': change, 'diabetes_med': med}
    #st.write(data)
    input = pd.DataFrame([list(data.values())], columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test', 'A1Ctest', 'time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency', 'change', 'diabetes_med'])
    st.write(input)
    print(input)
    print(input.dtypes)
    result = model.predict(input)
    st.write(result)
    st.write(result[0])
    
    if (result[0] == 1):
        st.write("The patient will most likely be readmitted")
    else:
        st.write("The patient likely will NOT be readmitted")
