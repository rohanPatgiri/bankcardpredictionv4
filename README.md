## Bank Card Type Prediction

- Link to the Web App : https://bankcardpredictionv4-1.streamlit.app/



### Flow of the project 
1.  Mention the Problem Statement
2.  Mention the steps used upto building and training the model
3.  Mention the steps used to create the Web app
4.  Mention the steps used to deploy the web app
---

### 1. Problem Statement
- There are 4 categories of cards in the given dataset : Blue, Silver, Gold, Platinum.
- Our goal is to predict  what kind of card a customer is likely to issue based on the information known about the customer
- The dataset contains 21 columns, including the dependent variable, which is the card category that we are trying to predict.
- The other 20 columns contains contains various information about the customer, based on which we will try to predict the type of card that a customer is likely to issue

---




---
## Building the Project 
### 2. Building the  ML model 
- Since the dataset contains both the features and labels, this is a scenario where supervised machine learning algorithms can be used.
- Did EDA to check for missing values etc.
- Encoded the Categorical Variables using Label Encoder from sk learn
- Used PCA to reduce the number of features
- Trained the model using Random Forest Classifier from sklearn
- Saved the model using Pickle

### 3. Building the User Interface (Web App)
- Created a function to take input from the user.
 This function contains the code to :
 1. take input from the user of all the 19 variables
 2. Store it in a dictionary
 3. Convert that dictionary into a Pandas Dataframe
 4. finally, it returns the data frame

...........
 - Below, we call the user_input_features() function and store the return Pandas Dataframe in the variable named "input_df" 
.........

- Next step is encoding.
............

- Next, we display the single row Pandas Dataframe of the data entered by the user

..........
- Then we load the trained models
- First we apply PCA to input_df to transform the user input data to the same number of parameters as it was used to train the model 
- Then we feed this modified data to predict the card type and its probability
-   prediction = load_clf.predict(final_input)
-   prediction_proba = load_clf.predict_proba(final_input)

...........
-  Finally we have the Code to display the predicted card type and its probability



---
### 4. Deploying te Web App
- Done on Streamlit by connecting Github repository. 
