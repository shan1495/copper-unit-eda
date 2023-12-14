import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
#Predictions....
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from sklearn.model_selection import GridSearchCV



st.markdown(":red[Making Machine to think like us!!]")
df1 = st.session_state['final_df']

X=df1[['quantity tons','new_status','item_type','application','thickness','width','country','customer','product_ref']]
df1['selling_price'] = df1['selling_price'].fillna(df1['selling_price'].mean())
y=df1['selling_price']
st.markdown(":blue[Decision Tree Regressor for Predicting Selling Price]")
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtr = DecisionTreeRegressor()
param_grid = {'max_depth': [2, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['sqrt', 'log2','auto']}
# gridsearchcv
grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
st.write("My Mode predicting on ",best_model)
st.write("My Model predict best on the ", grid_search.best_params_)
st.write("My Model's best score ", grid_search.best_score_)

st.markdown(":purple[New Samples]")
with st.form('selling-price-form'):
    tons = st.slider("Number of Tons",min_value=0.5,max_value=10.0,step=0.5)
    thickness = st.slider("Thickness ",min_value=0.5,max_value=10.0,step=0.5)
    width = st.slider("Width",min_value=10,max_value=100,step=10)
    status = st.radio("Status",options=["Won","Loss"])
    itemtype = st.selectbox("Item Type", options=df1.item_type.unique())
    ctry = st.selectbox("Select Country", [x for x in range(20,100)])
    appln = st.selectbox("Applications", [x for x in range(10,100)])
    cust = st.text_input("Enter the customer ID")
    prod = st.text_input("Product reference")
    submit = st.form_submit_button("Predutct Selling price")
    if submit:
        new_data = [[tons,1 if status=='Won' else 0,itemtype,appln,thickness,width,ctry,cust,prod]]
        dtr.fit(X,y)
        sprice_predit = dtr.predict(new_data)
        st.write("Expected Selling Price: ", sprice_predit)

st.markdown(":blue[Decision Tree Classifier for Predicting Status to Won or Loss]")
nX=df1[['quantity tons','selling_price','item_type','application','thickness','width','country','customer','product_ref']]
ny=df1['new_status']
X_train, X_test, y_train, y_test = train_test_split(nX, ny, test_size=0.2, random_state=42)

# decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy}")

st.markdown("Predict Prospect Leads to be a Customer or Not")
with st.form('status-pred'):
    tons = st.slider("Number of Tons",min_value=0.5,max_value=10.0,step=0.5)
    thickness = st.slider("Thickness ",min_value=0.5,max_value=10.0,step=0.5)
    width = st.slider("Width",min_value=10,max_value=100,step=10)
    sel_price = st.number_input("Selling Price")
    itemtype = st.selectbox("Item Type", options=df1.item_type.unique())
    ctry = st.selectbox("Select Country", [x for x in range(20,100)])
    appln = st.selectbox("Applications", [x for x in range(10,100)])
    cust = st.text_input("Enter the customer ID")
    prod = st.text_input("Product reference")
    submit = st.form_submit_button("Predutct Status")
    if submit:
        sample_data = [[tons,sel_price,itemtype,appln,thickness,width,ctry,cust,prod]]
        dtr.fit(X,y)
        status_pred = dtc.predict(sample_data)
        st.write("Expected Selling Price: ", "Won" if status_pred==1 else "Loss")



