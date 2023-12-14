import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import utils.utility as util

st.markdown("Sknewness finding")
st.write("Sknewness shows how the data is distributed. It should be normally distributed always.")
df = st.session_state['cleaned_dataframe']
st.write(df.select_dtypes('number').skew(axis=0))
st.write("With the above data, it is evident that the quantity tons, thickness, selling_price attributes are positive skewness. \n")
st.write("Skewness = 0: Then normally distributed.\nSkewness > 0: Then more weight in the left tail of the distribution.\n Skewness < 0: Then more weight in the right tail of the distribution.")

with st.form('graph-form'):
    grp = st.form_submit_button("View Graph")
    plt.figure(figsize=(17,13))
    #sb.histplot(data = df['thickness'], kde=True)
    option = st.selectbox("Select Atrribute",options=['selling_price','quantity tons','thickness','width'])
#     if option:
#         with st.spinner("We are generating graph. Wait"):
#             f,ax = plt.subplots()
#             print("Option -->",option)
#             sb.histplot(df,x=option, kde=True,ax=ax)      
#     #sb.histplot(data = df['quantity tons'], kde=True)   
#             st.write(f)
# # f,ax = plt.subplots()
# x=df[['quantity tons','application','thickness','width','selling_price','country','customer','item_type']].corr()
# sb.heatmap(x, annot=True, cmap="YlGnBu")
# st.write(f)
    cols = ['selling_price','quantity tons','thickness','width']
    fix = st.form_submit_button('Fix Skewness')
    if fix:
        df = util.test_log_transforms(cols,df)
        st.session_state['final_df'] = df

    st.markdown("After fixing the skewing issue.. Now")
    st.write(df.select_dtypes('number').skew(axis=0))
