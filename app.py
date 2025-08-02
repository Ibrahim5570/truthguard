import streamlit as st
import pandas as pd
import numpy as np

st.title("📦 Dependency Check")

# Confirm if pandas is working
st.write("✅ pandas version:", pd.__version__)
st.write("✅ numpy version:", np.__version__)
