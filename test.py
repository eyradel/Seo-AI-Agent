import streamlit as st

import google.generativeai as genai

genai.configure(api_key="AIzaSyBaw_lRd8A7l9X1oQfwLRzKC97LH3UcRh8")

model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Explain how AI works")

st.write(response.text)