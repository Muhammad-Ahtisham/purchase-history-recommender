import streamlit as st
import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# ------------------ SETUP ------------------
st.set_page_config(page_title="Product Recommendation", layout="centered")
st.title("🔍 Surgical Tool Recommendation System")

# ---------- DATABASE CONNECTION ----------
conn = sqlite3.connect("recommendation.db", check_same_thread=False)
cursor = conn.cursor()

# ---------- INIT DB TABLES ----------
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    userID TEXT PRIMARY KEY,
    previousPurchases TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS tools (
    Title TEXT,
    Title_URL TEXT,
    Image TEXT
)''')
conn.commit()

# ---------- LOAD DATA FROM DATABASE ----------
@st.cache_data(show_spinner=False)
def load_data_fresh():
    user_df = pd.read_sql_query("SELECT * FROM users", conn)
    tools_df = pd.read_sql_query("SELECT * FROM tools", conn)
    return user_df, tools_df

# ---------- FIND MATCH FUNCTION ----------
def find_best_match(prod_name, choices, threshold=70):
    match, score = process.extractOne(prod_name.lower().strip(), choices)
    if score >= threshold:
        return match
    return None

# ---------- DATA PIPELINE FUNCTION ----------
def get_updated_data():
    df, tools_df = load_data_fresh()
    purchase_matrix = df.set_index('userID')['previousPurchases'].str.get_dummies(sep='|')
    sim_matrix = cosine_similarity(purchase_matrix.values)
    sim_df = pd.DataFrame(sim_matrix, index=purchase_matrix.index, columns=purchase_matrix.index)
    tools_df['Title_clean'] = tools_df['Title'].str.lower().str.strip()
    product_choices = tools_df['Title_clean'].tolist()
    return df, tools_df, purchase_matrix, sim_df, product_choices

# ---------- TABS ----------
tab1, tab2 = st.tabs(["📊 Recommend Products", "➕ Add New User"])

# ========== TAB 1: RECOMMENDATION ==========
with tab1:
    df, tools_df, purchase_matrix, sim_df, product_choices = get_updated_data()
    st.write("## 📌 User-Based Product Recommendations")
    user_list = list(purchase_matrix.index)
    selected_user = st.selectbox("Select a User ID", user_list)
    custom_user_input = st.text_input("Or enter a User ID manually:", value=selected_user)

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
                st.subheader("🎯 Top 5 Recommended Products:")
                for prod in top5.index:
                    best_match = find_best_match(prod, product_choices)
                    if best_match:
                        row = tools_df[tools_df['Title_clean'] == best_match].iloc[0]
                        st.markdown(f"### [{prod}]({row['Title_URL']})")
                        try:
                            st.image(row['Image'], use_container_width=True)
                        except:
                            st.write("(Image unavailable)")
                    else:
                        st.write(f"- {prod} (No match found)")
    else:
        st.warning("User ID not found in the dataset.")

# ========== TAB 2: ADD NEW USER ==========
with tab2:
    st.write("## ➕ Create a New User Profile")
    new_user_id = st.text_input("🔹 Enter New User ID")
    new_user_purchases = st.text_input("🔹 Purchased tools (use '|' to separate multiple items):")

    if st.button("✅ Add User and Generate Recommendations"):
        if new_user_id.strip() == "" or new_user_purchases.strip() == "":
            st.warning("Please enter both User ID and purchase history.")
        else:
            # Check if user already exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE userID=?", (new_user_id,))
            if cursor.fetchone()[0] > 0:
                st.warning("User ID already exists. Please choose another one.")
            else:
                cursor.execute("INSERT INTO users (userID, previousPurchases) VALUES (?, ?)",
                               (new_user_id.strip(), new_user_purchases.strip()))
                conn.commit()
                st.success(f"User '{new_user_id}' added successfully!")
                st.cache_data.clear()

                # Reload and recommend
                df, tools_df, purchase_matrix, sim_df, product_choices = get_updated_data()
                sim_scores = sim_df[new_user_id].drop(new_user_id)
                sim_scores = sim_scores[sim_scores > 0]

                if sim_scores.empty:
                    st.info("No similar users found. Showing most popular tools instead.")
                    top_products = purchase_matrix.sum().sort_values(ascending=False).head(5)
                else:
                    weighted_scores = purchase_matrix.loc[sim_scores.index].T.dot(sim_scores)
                    user_vector = purchase_matrix.loc[new_user_id]
                    new_scores = weighted_scores[user_vector == 0]
                    top_products = new_scores.sort_values(ascending=False).head(5)

                st.subheader(f"🎁 Top 5 Recommendations for {new_user_id}:")
                for prod in top_products.index:
                    best_match = find_best_match(prod, product_choices)
                    if best_match:
                        row = tools_df[tools_df['Title_clean'] == best_match].iloc[0]
                        st.markdown(f"### [{prod}]({row['Title_URL']})")
                        try:
                            st.image(row['Image'], use_container_width=True)
                        except:
                            st.write("(Image unavailable)")
                    else:
                        st.write(f"- {prod} (No match found)")
