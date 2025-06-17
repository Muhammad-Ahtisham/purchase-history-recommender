import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# --- Setup ---
st.set_page_config(page_title="Product Recommendation", layout="centered")
st.title("User-Based Product Recommendation")
st.write("Product recommendations based on purchase history similarity.")

# --- Load Data from SQLite ---
@st.cache_data
def load_data():
    conn = sqlite3.connect("recommendation.db")
    users_df = pd.read_sql_query("SELECT * FROM users", conn)
    tools_df = pd.read_sql_query("SELECT * FROM tools", conn)
    conn.close()
    return users_df, tools_df

try:
    df, tools_df = load_data()
except Exception as e:
    st.error(f"❌ Error loading database: {e}")
    st.stop()

# Check for required columns
if 'userID' not in df.columns or 'previousPurchases' not in df.columns:
    st.error("The 'users' table must contain 'userID' and 'previousPurchases'.")
    st.stop()

# --- Preprocessing ---
purchase_matrix = df.set_index('userID')['previousPurchases'].str.get_dummies(sep='|')
sim_matrix = cosine_similarity(purchase_matrix.values)
sim_df = pd.DataFrame(sim_matrix, index=purchase_matrix.index, columns=purchase_matrix.index)

tools_df['Title_clean'] = tools_df['Title'].str.lower().str.strip()
product_choices = tools_df['Title_clean'].tolist()

def find_best_match(prod_name, choices, threshold=70):
    match, score = process.extractOne(prod_name.lower().strip(), choices)
    return match if score >= threshold else None

# --- User Selection ---
st.write("### Select or Input a User to Recommend Products")
user_list = list(purchase_matrix.index)
selected_user = st.selectbox("Select a User ID", user_list)
custom_user_input = st.text_input("Or enter a User ID manually:", value=selected_user)

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
            for prod in top5.index:
                best_match = find_best_match(prod, product_choices)
                if best_match:
                    row = tools_df[tools_df['Title_clean'] == best_match].iloc[0]
                    st.markdown(f"### [{prod}]({row['Title_URL']})")
                    try:
                        st.image(row['Image'], use_container_width=True)
                    except:
                        st.write(f"🖼️ [Image not available]({row['Image']})")
                else:
                    st.write(f"– {prod} (No image found)")
else:
    st.warning("User ID not found in the dataset.")

# --- Add New User ---
st.write("---")
st.write("## 🆕 Add New User Profile")

new_user_id = st.text_input("🔹 Enter New User ID")
new_user_purchases = st.text_input("🔹 Purchased tools (use '|' to separate multiple items):")

if st.button("✅ Add and Recommend"):
    if not new_user_id.strip():
        st.warning("Please enter a valid User ID.")
    elif new_user_id in purchase_matrix.index:
        st.warning("User already exists.")
    else:
        conn = sqlite3.connect("recommendation.db")
        try:
            conn.execute(
                "INSERT INTO users (userID, previousPurchases) VALUES (?, ?)",
                (new_user_id, new_user_purchases.strip())
            )
            conn.commit()
            st.success(f"User '{new_user_id}' added!")

            # Reload data
            df, tools_df = load_data()
            purchase_matrix = df.set_index('userID')['previousPurchases'].str.get_dummies(sep='|')
            sim_matrix = cosine_similarity(purchase_matrix.values)
            sim_df = pd.DataFrame(sim_matrix, index=purchase_matrix.index, columns=purchase_matrix.index)

            if new_user_purchases.strip():
                sim_scores = sim_df[new_user_id].drop(new_user_id)
                sim_scores = sim_scores[sim_scores > 0]

                if sim_scores.empty:
                    st.write("No similar users.")
                else:
                    weighted_scores = purchase_matrix.loc[sim_scores.index].T.dot(sim_scores)
                    user_vector = purchase_matrix.loc[new_user_id]
                    new_scores = weighted_scores[user_vector == 0]
                    top5 = new_scores.sort_values(ascending=False).head(5)

                    if top5.empty:
                        st.write("No recommendations yet.")
                    else:
                        st.subheader(f"Top 5 Recommendations for {new_user_id}:")
                        for prod in top5.index:
                            best_match = find_best_match(prod, product_choices)
                            if best_match:
                                row = tools_df[tools_df['Title_clean'] == best_match].iloc[0]
                                st.markdown(f"### [{prod}]({row['Title_URL']})")
                                try:
                                    st.image(row['Image'], use_container_width=True)
                                except:
                                    st.write(f"🖼️ [Image not available]({row['Image']})")
                            else:
                                st.write(f"– {prod} (No image found)")
            else:
                st.subheader("Default Recommendations (Popular Tools):")
                top_products = purchase_matrix.sum().sort_values(ascending=False).head(5)
                for prod in top_products.index:
                    best_match = find_best_match(prod, product_choices)
                    if best_match:
                        row = tools_df[tools_df['Title_clean'] == best_match].iloc[0]
                        st.markdown(f"### [{prod}]({row['Title_URL']})")
                        try:
                            st.image(row['Image'], use_container_width=True)
                        except:
                            st.write(f"🖼️ [Image not available]({row['Image']})")
        except Exception as e:
            st.error(f"Database error: {e}")
        finally:
            conn.close()
