import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

st.markdown("A picture speaks more than a Thousand words")
st.write("Data visualization is one the important steps in the :green[E]xpoloratory :green[D]ata :green[A]nalysis")
df = st.session_state['cleaned_dataframe']
print("Got the Data set from session", df.size)
st.write("The Correlation of target variable")
corr = df.select_dtypes('number').corr()
figure,ax = plt.subplots()
sb.heatmap(corr,annot=True,fmt=".2f",ax=ax)
st.write(figure)
st.write("Outliars for the data set")
fig,ax = plt.subplots()
sb.boxplot(data=df,y=df['selling_price'],ax=ax)
#plt.figure(fig)
st.write(fig)
fig,ax = plt.subplots()
sb.violinplot(y=df['quantity tons'],ax=ax)
#plt.figure(fig)
st.write(fig)

