import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up the Streamlit app
st.set_page_config(page_title="Product Recommendation", layout="centered")
st.title("User-Based Product Recommendation")
st.write("Product recommendations based on purchase history similarity.")

# Load user purchase history dataset
@st.cache_data
def load_data():
    return pd.read_excel("surgical_tool_recommendation_users.xlsx")

# Load tool dataset and extract product code from title
@st.cache_data
def load_tool_data():
    df_tool = pd.read_excel("Tools_1.xlsx")
    df_tool['ProductID'] = df_tool['Title'].astype(str).str.extract(r'\|\s*(.+)$')[0]
    df_tool['ProductID'] = df_tool['ProductID'].str.strip().str.lower()
    df_tool['ImageURL'] = df_tool['Image'].astype(str).fillna('').str.strip()
    return df_tool

# Load both datasets
try:
    df = load_data()
    tool_df = load_tool_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

# Validate user data columns
if 'userID' not in df.columns or 'previousPurchases' not in df.columns:
    st.error("The dataset must contain 'userID' and 'previousPurchases' columns.")
    st.stop()

# Validate tool data columns
if 'ProductID' not in tool_df.columns or 'ImageURL' not in tool_df.columns:
    st.error("The tool dataset must contain 'ProductID' and 'ImageURL' columns.")
    st.stop()

# One-hot encode the 'previousPurchases' by splitting on '|'
purchase_matrix = df.set_index('userID')['previousPurchases'].str.get_dummies(sep='|')
purchase_matrix.columns = purchase_matrix.columns.str.strip().str.lower()  # Normalize column names

# Compute cosine similarity between users
sim_matrix = cosine_similarity(purchase_matrix.values)
sim_df = pd.DataFrame(sim_matrix, index=purchase_matrix.index, columns=purchase_matrix.index)

st.success("Data loaded successfully. Similarity matrix computed.")
st.write("### Select or Input a User to Recommend Products")

# User selection
user_list = list(purchase_matrix.index)
selected_user = st.selectbox("Select a User ID", user_list)
custom_user_input = st.text_input("Or enter a User ID manually:", value=selected_user)

# Function to get image URL by product ID (normalized to lowercase)
def get_image_url(product_id):
    product_id = product_id.strip().lower()
    match = tool_df[tool_df['ProductID'] == product_id]
    if not match.empty:
        url = match['ImageURL'].values[0]
        return url if url.startswith("http") else None
    return None

# Recommend
if custom_user_input in purchase_matrix.index:
    selected_user = custom_user_input
    sim_scores = sim_df[selected_user].drop(selected_user)
    sim_scores = sim_scores[sim_scores > 0]

    if sim_scores.empty:
        st.write("No similar users found for this user.")
    else:
        weighted_scores = purchase_matrix.loc[sim_scores.index].T.dot(sim_scores)
        user_vector = purchase_matrix.loc[selected_user]
        new_scores = weighted_scores[user_vector == 0]
        top5 = new_scores.sort_values(ascending=False).head(5)

        if top5.empty:
            st.write("No new product recommendations available for this user.")
        else:
            st.subheader("Top 5 Recommended Products:")
            for prod in top5.index:
                st.markdown(f"**{prod}**")
                img_url = get_image_url(prod)
                if img_url:
                    st.image(img_url, width=200)
                else:
                    st.warning("🔍 Image not available.")
else:
    st.warning("User ID not found in the dataset.")
