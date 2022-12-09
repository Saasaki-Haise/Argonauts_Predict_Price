# Import
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st


st.write('''
    # Argonauts Sale Price Prediction
    Predict best price for sale you'r Argonauts
''')

st.sidebar.header("Data Entry Parameters")

def user_input():
    nftId = st.sidebar.slider('NFD #ID | ( enter argonauts ID ) : ', 1, 8888, 1)
    rank = st.sidebar.slider('Rank | ( enter argonauts rank ) : ', 1, 8888, 1)
    traitCount = st.sidebar.slider('Trait Count | ( enter argonauts traits counts ) : ', 1, 8, 8)
    archetype = st.sidebar.slider('Archetype (1 = Spirit | 2 = Galaxy | 3 = Celestial | 4 = Terrestrial):', 1, 4, 4)
    floorPrice = st.sidebar.slider('Floor Price | ( enter the actual floor price ) : ', 0, 10000, 1000)
    data = {
        'nftId':nftId,
        'Rank':rank,
        'Trait Count':traitCount,
        'Archetype':archetype,
        'Floor Price':floorPrice
    }
    argonauts_param = pd.DataFrame(data, index=[0])
    return argonauts_param

argonauts=user_input()

st.subheader("I want predict price this Argonauts :")
st.write(argonauts)

df = pd.read_json('./data/testClean.json')
df_x = pd.DataFrame(df , columns=['nftId','Rank','Trait Count','Archetype','Floor Price'])
df_y = pd.DataFrame(df.price)
reg = linear_model.LinearRegression()
X_train,X_test, y_train,y_test = train_test_split(df_x,df_y, test_size=0.2, random_state=3)
reg.fit(X_train,y_train)

prediction = reg.predict(argonauts)

st.subheader('Best price is :')
st.write(prediction)
