import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))

def predict(g, m, d, e, se, ai, ci, la, lat, ch, pa):
    predict_df = pd.DataFrame({"Gender": [g], "Married": [m], "Dependents": [d], "Education": [e], "Self_Employed": [se], 
                        "ApplicantIncome": [ai], "CoapplicantIncome": [ci], "LoanAmount": [la], "Loan_Amount_Term": [lat],
                        "Credit_History": [ch], "Property_Area": [pa]})
    prediction = model.predict(predict_df)
    return prediction


def main():
    st.title("Loan Approval Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style = "color:white;text-align:center;">Streamlit Loan Approval Predictor </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    g = st.text_input("Gender", "Type Here")
    m = st.text_input("Married", "Type Here")
    d = st.text_input("Dependents", "Type Here")
    e = st.text_input("Education", "Type Here")
    se = st.text_input("Self_Employed", "Type Here")
    ai = st.text_input("Applicant Income", "Type Here")
    ci = st.text_input("CoApplicant Income", "Type Here")
    la = st.text_input("Loan Amount", "Type Here")
    lat = st.text_input("Loan Amount term", "Type Here")
    ch = st.text_input("Credit History", "Type Here")
    pa = st.text_input("Preoperty Area", "Type Here")

    result = ""
    if st.button("Predict"):
        result = predict(g, m, d, e, se, ai, ci, la, lat, ch, pa)
    st.success("The output is {}".format(result))
    if st.button("About"):
        st.text("Predicts the Approval of Loan")
        st.text("Built with Streamlit")

main()