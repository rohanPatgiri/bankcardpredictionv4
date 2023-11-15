import streamlit as st
import pandas as pd
import pickle
import numpy as np
st.write ("## Card Type prediction App ")



st.sidebar.header("Enter the user data below")

# This function contains the code to :
# 1. take input from the user of all the 19 variables
# 2. Store it in a dictionary
# 3. Convert that dictionary into a Pandas Dataframe
# 4. finally, it returns the data frame
def user_input_features():
   Attrition_Flag  = st.sidebar.selectbox('Attrition',('Existing Customer','Attrited Customer'))
   Education_Level = st.sidebar.selectbox('Education Level',("College","Doctorate","Graduate","High School","Post-Graduate","Uneducated","Unknown"))
   Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
   Marital_Status = st.sidebar.selectbox("Marital Status", ("Divorced", "Married", "Single", "Unknown"))
   Income_Category = st.sidebar.selectbox("Income Category",("$120K +", "$40K - $60K", "$60K - $80K", "$80K - $120K", "Less than $40K", "Unknown"))

   Customer_Age = st.sidebar.slider("Customer Age",25,70,32)
   Dependent_count =   st.sidebar.slider("Dependent Counnt",0,6,2)
   Months_on_book =  st.sidebar.slider("Months on Book",10,80,32)
   Total_Relationship_Count =  st.sidebar.slider("Total Relationship Count",0,8,1)
   Months_Inactive_12_mon =  st.sidebar.slider("Months Inactive",0,12,2)

   Contacts_Count_12_mon =  st.sidebar.slider("Contacts Count",0,10,7)
   Credit_Limit =  st.sidebar.slider("Credit Limit",1000,50000,5000)
   Total_Revolving_Bal = st.sidebar.slider("Total Revolving Balance",0,5000,1000)
   Avg_Open_To_Buy =  st.sidebar.slider("Average Open to Buy",0,20,18)
   Total_Amt_Chng_Q4_Q1 =  st.sidebar.slider("Total Amount Change (Q1 to Q4)",0,50000,20000)

   Total_Trans_Amt =  st.sidebar.slider("Total Transaction Amount",0,5000,1000)
   Total_Trans_Ct = st.sidebar.slider("Total Tansaction Cost",0,100,20)
   Total_Ct_Chng_Q4_Q1 =  st.sidebar.slider("Total Cost Change (Q1 to Q4)",0,4,1)
   Avg_Utilization_Ratio =  st.sidebar.slider("Average Utilization Ratio",0,100, 80)
   Avg_Utilization_Ratio = Avg_Utilization_Ratio/100
   
   data = {"Attrition_Flag": Attrition_Flag,
           "Customer_Age": Customer_Age,
           "Gender": Gender,
           "Dependent_count": Dependent_count,
           "Education_Level": Education_Level,
           "Marital_Status": Marital_Status,
           "Income_Category" : Income_Category,
           "Months_on_book": Months_on_book,
           "Total_Relationship_Count": Total_Relationship_Count,
           "Months_Inactive_12_mon": Months_Inactive_12_mon,
           "Contacts_Count_12_mon":Contacts_Count_12_mon,
           "Credit_Limit": Credit_Limit,
           "Total_Revolving_Bal": Total_Revolving_Bal,
           "Avg_Open_To_Buy": Avg_Open_To_Buy,
           "Total_Amt_Chng_Q4_Q1": Total_Amt_Chng_Q4_Q1,
           "Total_Trans_Amt": Total_Trans_Amt,
           "Total_Trans_Ct": Total_Trans_Ct,
           "Total_Ct_Chng_Q4_Q1": Total_Ct_Chng_Q4_Q1,
           "Avg_Utilization_Ratio": Avg_Utilization_Ratio}
   features = pd.DataFrame(data,index = [0])
   return features

# Below, we call the user_input_features() function and store the return Pandas Dataframe...
# ... in the variable named "input_df" 
input_df = user_input_features()

#...........
#  Code for Encoding ctegorical variables which contains two categories
def ref1(x):
    if x=="Male":
        return 1
    else:
        return 0
    
input_df["Gender"] = input_df["Gender"].map(ref1)


def ref2(x):
    if x=="Existing Customer":
        return 1
    else:
        return 0
    
input_df["Attrition_Flag"] = input_df["Attrition_Flag"].map(ref2)


# this is the code for converting categorical variables which has more than two categories
from sklearn.preprocessing import LabelEncoder

def label_encoded(feat):
    le = LabelEncoder()
    le.fit(feat)
    print(feat.name, le.classes_)
    return le.transform(feat)

input_df["Income_Category"]= label_encoded(input_df['Income_Category'])
input_df["Education_Level"]= label_encoded(input_df['Education_Level'])
input_df["Marital_Status"]= label_encoded(input_df['Marital_Status'])

#  Code for Encoding ends


#........................

# Displays the user input features
st.subheader('User Input features')
# This code displays the single row Pandas Dataframe of the data entered by the user
st.write(input_df)

# This code is ued to load the saved models 
load_clf = pickle.load(open('BankCards.pickle', 'rb'))
scalar = pickle.load(open('BankCardsPCA.pickle','rb'))

# First we apply PCA to input_df to transform it to the same number of parameters as it was used to train the model 
final_input = scalar.transform(np.array(input_df).reshape(1,-1))


# Apply model to make predictions
prediction = load_clf.predict(final_input)
prediction_proba = load_clf.predict_proba(final_input)


st.subheader('Prediction')
#penguins_species = np.array(['Blue','Gold','Silver', 'Platinum'])
#st.write(penguins_species[prediction])

# Code to display the predicted card type and its probability

st.subheader('Prediction Probability')
st.write(prediction_proba)
st.write("The predicted Card type is: ", prediction)