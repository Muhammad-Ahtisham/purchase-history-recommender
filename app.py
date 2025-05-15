import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Set up the Streamlit app
st.set_page_config(page_title="Product Recommendation", layout="centered")
st.title("User-Based Product Recommendation")
st.write("Product recommendations based on purchase history similarity.")

# Load dataset automatically
@st.cache_data
def load_user_data():
    return pd.read_excel("surgical_tool_recommendation_users.xlsx")

@st.cache_data
def load_product_data():
    return pd.read_excel("Tools_1.xlsx")

try:
    df = load_user_data()
    tools_df = load_product_data()
except Exception as e:
    st.error(f"Error loading the dataset: {e}")
    st.stop()

# Ensure required columns exist
if 'userID' not in df.columns or 'previousPurchases' not in df.columns:
    st.error("The dataset must contain 'userID' and 'previousPurchases' columns.")
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

# Prepare cleaned product titles for matching
tools_df['Title_clean'] = tools_df['Title'].str.lower().str.strip()
product_choices = tools_df['Title_clean'].tolist()

def find_best_match(prod_name, choices, threshold=70):
    match, score = process.extractOne(prod_name.lower().strip(), choices)
    if score >= threshold:
        return match
    return None

# Allow user to either select or enter a user ID
user_list = list(purchase_matrix.index)
selected_user = st.selectbox("Select a User ID", user_list)
custom_user_input = st.text_input("Or enter a User ID manually:", value=selected_user)

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
                best_match = find_best_match(prod, product_choices)

                if best_match:
                    row = tools_df[tools_df['Title_clean'] == best_match].iloc[0]
                    image_url = row['Image']
                    title_url = row['Title_URL']
                    st.markdown(f"### [{prod}]({title_url})")
                    st.image(image_url, use_column_width=True)
                else:
                    st.write(f"– {prod} (No image found)")
else:
    st.warning("User ID not found in the dataset.")
