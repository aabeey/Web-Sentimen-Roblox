import streamlit as st
import pickle

with open("model_nb.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📊 Web Analisis Sentimen")

text_input = st.text_area("Masukkan teks komentar:", height=150)

if st.button("Analisis"):
    if text_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        vec = vectorizer.transform([text_input])
        pred = model.predict(vec)

        if pred[0] == 1:
            st.success("Sentimen: Positif")
        else:
            st.error("Sentimen: Negatif")

# ==== STYLE ====
st.markdown("""
<style>
h2 { text-align: center; white-space: nowrap; }
div.stButton > button,
div.stButton > button:hover,
div.stButton > button:active, 
div.stButton > button:focus {
    background-color: #8e918f !important;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
}
.result {
     font-size: 20px;
     font-weight: bold;
     text-align: center;
     margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)
