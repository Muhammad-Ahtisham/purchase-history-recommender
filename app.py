import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page setup
st.set_page_config(page_title="Product Recommendation", layout="centered")
st.title("User-Based Product Recommendation")

# Load user data
@st.cache_data
def load_data():
    return pd.read_excel("surgical_tool_recommendation_users.xlsx")

# Load tools data and extract ProductID
@st.cache_data
def load_tool_data():
    df_tool = pd.read_excel("/mnt/data/Tools_1.xlsx")
    # Extract ProductID from title
    df_tool['ProductID'] = df_tool['title'].astype(str).str.extract(r'\|\s*(.+)$')[0]
    df_tool['ProductID'] = df_tool['ProductID'].str.strip().str.lower()
    df_tool['ImageURL'] = df_tool['ImageURL'].astype(str).fillna('').str.strip()
    return df_tool

try:
    df = load_data()
    tool_df = load_tool_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

# Validate columns
if 'userID' not in df.columns or 'previousPurchases' not in df.columns:
    st.error("User data must contain 'userID' and 'previousPurchases'.")
    st.stop()

if 'ProductID' not in tool_df.columns or 'ImageURL' not in tool_df.columns:
    st.error("Tool data must contain 'ProductID' and 'ImageURL'.")
    st.stop()

# One-hot encode purchases
purchase_matrix = df.set_index('userID')['previousPurchases'].str.get_dummies(sep='|')
purchase_matrix.columns = purchase_matrix.columns.str.strip().str.lower()

# Similarity matrix
sim_matrix = cosine_similarity(purchase_matrix.values)
sim_df = pd.DataFrame(sim_matrix, index=purchase_matrix.index, columns=purchase_matrix.index)

# UI Input
user_list = list(purchase_matrix.index)
selected_user = st.selectbox("Select a User ID", user_list)
custom_user_input = st.text_input("Or enter a User ID manually:", value=selected_user)

# Match function
def get_image_url(product_id):
    product_id = product_id.strip().lower()
    match = tool_df[tool_df['ProductID'] == product_id]
    if not match.empty:
        url = match['ImageURL'].values[0]
        return url if url.startswith("http") else None
    return None

# Recommendation engine
if custom_user_input in purchase_matrix.index:
    selected_user = custom_user_input
    sim_scores = sim_df[selected_user].drop(selected_user)
    sim_scores = sim_scores[sim_scores > 0]

    if sim_scores.empty:
        st.write("No similar users found.")
    else:
        weighted_scores = purchase_matrix.loc[sim_scores.index].T.dot(sim_scores)
        user_vector = purchase_matrix.loc[selected_user]
        new_scores = weighted_scores[user_vector == 0]
        top5 = new_scores.sort_values(ascending=False).head(5)

        if top5.empty:
            st.write("No new product recommendations.")
        else:
            st.subheader("Top 5 Recommended Products:")
            
            debug_table = []

            for prod in top5.index:
                normalized_prod = prod.strip().lower()
                img_url = get_image_url(normalized_prod)
                
                debug_table.append({
                    "Recommended Product": prod,
                    "Normalized ID": normalized_prod,
                    "Image Found": "✅" if img_url else "❌",
                    "ImageURL": img_url if img_url else "N/A"
                })
                
                st.markdown(f"**{prod}**")
                if img_url:
                    st.image(img_url, width=200)
                else:
                    st.warning("🔍 Image not available.")

            # Debug info
            st.write("### Debug Info: Recommendation Matching")
            st.dataframe(pd.DataFrame(debug_table))

            # Optional: print full set of extracted product IDs from tool file
            with st.expander("Show all available tool ProductIDs"):
                st.write(tool_df['ProductID'].unique())
else:
    st.warning("User ID not found in the dataset.")
