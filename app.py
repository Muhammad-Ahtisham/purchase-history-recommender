import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up the Streamlit app
st.set_page_config(page_title="Product Recommendation", layout="centered")
st.title("User-Based Product Recommendation")
st.write("Upload purchase history data to get product recommendations.")

# File uploader for Excel files
uploaded_file = st.file_uploader("Upload an Excel file (purchase history)", type=["xlsx", "xls"])
if uploaded_file is not None:
    try:
        # Read the uploaded Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    # Ensure required columns exist
    if 'userID' not in df.columns or 'previousPurchases' not in df.columns:
        st.error("The file must contain 'userID' and 'previousPurchases' columns.")
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
                    st.write("–", prod)
    else:
        st.warning("User ID not found in the uploaded data.")
else:
    st.info("🔄 Please upload a purchase history Excel file to begin.")
