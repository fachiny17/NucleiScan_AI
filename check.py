import streamlit as st
import os

# Debug check
st.write("Secrets available:", list(st.secrets.keys()))
st.write("Firebase creds exists:", "FIREBASE_CREDENTIALS_B64" in st.secrets)