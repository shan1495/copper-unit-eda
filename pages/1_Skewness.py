import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

st.markdown("Sknewness finding")
st.write("Sknewness shows how the data is distributed. It should be normally distributed always.")
with st.form(key='load-data'):
    clicked = st.form_submit_button(label='Load Data')
    if clicked:
        with st.spinner("Loading the data"):
            
            df = pd.read_excel('.\\dataset\\Copper_Set.xlsx',dtype='unicode')
           
            #df = df.groupby(['delivery date']).count().reset_index()
        fig,ax = plt.subplots()
        df = df.drop(df.query('item_date == "19950000"').index)
        print("Deleting the 19950000 data .. and 1919 data...")
        df = df.drop(df.query('item_date== "20191919"').index)
        df = df.drop(df.query('`delivery date` == "30310101"').index)
        df = df.drop(df.query('`delivery date` == "20212222"').index)
        df = df.drop(df.query('status !="Won" & status !="Loss"').index)
        st.dataframe(data=df)
        df['item_date']=pd.to_datetime(df['item_date'], format='mixed').dt.date
        df['delivery date']=pd.to_datetime(df['delivery date'],format='mixed').dt.date
        df.drop('material')
            
        sb.histplot(data=df,x='delivery date',binwidth=10,kde=True,bins=20, label="Skewness: %.2f"%(df["selling_price"].skew()))
        df['country'].mode()
        df['country'].fill
        #sb.lineplot(data=df,x='delivery date',y='selling_price')
        plt.figure(fig)
        st.write(fig)