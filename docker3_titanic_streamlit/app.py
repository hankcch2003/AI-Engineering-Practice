import streamlit as st

# Title (標題)
st.title("鐵達尼號乘客生存預測系統")

# User input (使用者輸入)
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0, 500, 50)

# Prediction button (預測按鈕)
if st.button("Predict"):
    if age < 18:
        st.success("Survived (likely)")
    else:
        st.error("Not Survived (likely)")