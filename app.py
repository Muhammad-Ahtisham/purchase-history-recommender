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

# Load tool image and product ID dataset
@st.cache_data
def load_tool_data():
    return pd.read_excel("/mnt/data/Tools_1.xlsx")

# Try loading both datasets
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading the user dataset: {e}")
    st.stop()

try:
    tool_df = load_tool_data()
except Exception as e:
    st.error(f"Error loading the tool dataset: {e}")
    st.stop()

# Check required columns in user dataset
if 'userID' not in df.columns or 'previousPurchases' not in df.columns:
    st.error("The dataset must contain 'userID' and 'previousPurchases' columns.")
    st.stop()

# Check required columns in tool dataset
if 'ProductID' not in tool_df.columns or 'ImageURL' not in tool_df.columns:
    st.error("The tool dataset must contain 'ProductID' and 'ImageURL' columns.")
    st.stop()

# One-hot encode the 'previousPurchases' by splitting on '|'
purchase_matrix = df.set_index('userID')['previousPurchases'].str.get_dummies(sep='|')

# Compute cosine similarity between users
sim_matrix = cosine_similarity(purchase_matrix.values)
sim_df = pd.DataFrame(sim_matrix,
                      index=purchase_matrix.index,
                      columns=purchase_matrix.index)

st.success("Data loaded successfully. Similarity matrix computed.")
st.write("### Select or Input a User to Recommend Products")

# Allow user to either select or enter a user ID
user_list = list(purchase_matrix.index)
selected_user = st.selectbox("Select a User ID", user_list)
custom_user_input = st.text_input("Or enter a User ID manually:", value=selected_user)

# Function to get image URL by product name
def get_image_url(product_name):
    match = tool_df[tool_df['ProductID'].astype(str).str.lower() == product_name.lower()]
    if not match.empty:
        return match['ImageURL'].values[0]
    return None

if custom_user_input in purchase_matrix.index:
    selected_user = custom_user_input

    # Similarity scores for the selected user (exclude self, only positive)
    sim_scores = sim_df[selected_user].drop(selected_user)
    sim_scores = sim_scores[sim_scores > 0]

    if sim_scores.empty:
        st.write("No similar users found for this user.")
    else:
        # Weighted sum of product vectors of similar users
        weighted_scores = purchase_matrix.loc[sim_scores.index].T.dot(sim_scores)
        # Remove products already bought by the selected user
        user_vector = purchase_matrix.loc[selected_user]
        new_scores = weighted_scores[user_vector == 0]
        # Get top 5 recommendations
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
                    st.write("🔍 Image not available.")
else:
    st.warning("User ID not found in the dataset.")
