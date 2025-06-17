import streamlit as st
import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------ SETUP ------------------
st.set_page_config(page_title="Product Recommendation", layout="centered")
st.title("🔍 Surgical Tool Recommendation System")

# ---------- DATABASE CONNECTION ----------
conn = sqlite3.connect("recommendation.db", check_same_thread=False)
cursor = conn.cursor()

# ---------- DROP AND CREATE USERS TABLE ----------
cursor.execute("DROP TABLE IF EXISTS users")
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    userID TEXT PRIMARY KEY,
    previousPurchases TEXT,
    specialty TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS tools (
    Title TEXT,
    Title_URL TEXT,
    Image TEXT,
    Category TEXT,
    Specialty TEXT
)''')
conn.commit()

# ---------- LOAD DATA ----------
@st.cache_data(show_spinner=False)
def load_data_fresh():
    user_df = pd.read_sql_query("SELECT * FROM users", conn)
    tools_df = pd.read_sql_query("SELECT * FROM tools", conn)
    tools_df.columns = tools_df.columns.str.strip().str.lower()
    return user_df, tools_df

# ---------- FUZZY MATCH ----------
def find_best_match(prod_name, choices, threshold=70):
    match, score = process.extractOne(prod_name.lower().strip(), choices)
    if score >= threshold:
        return match
    return None

# ---------- DATA PIPELINE ----------
def get_updated_data():
    df, tools_df = load_data_fresh()
    if df.empty:
        return df, tools_df, pd.DataFrame(), pd.DataFrame(), []

    purchase_matrix = df.set_index('userID')['previousPurchases'].str.get_dummies(sep='|')
    sim_matrix = cosine_similarity(purchase_matrix.values)
    sim_df = pd.DataFrame(sim_matrix, index=purchase_matrix.index, columns=purchase_matrix.index)
    tools_df['title_clean'] = tools_df['title'].str.lower().str.strip()
    product_choices = tools_df['title_clean'].tolist()
    return df, tools_df, purchase_matrix, sim_df, product_choices

# ---------- CONTENT-BASED FILTERING ----------
def similar_products_view(tools_df, selected_title):
    tfidf = TfidfVectorizer(stop_words='english')
    specs_matrix = tfidf.fit_transform(tools_df['category'].fillna(''))
    idx = tools_df[tools_df['title'] == selected_title].index[0]
    cosine_sim = cosine_similarity(specs_matrix[idx], specs_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-6:][::-1]
    return tools_df.iloc[similar_indices[1:]]

def products_by_specialty(tools_df, specialty):
    return tools_df[tools_df['category'].str.lower() == specialty.lower()].head(5)

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["📊 Recommend Products", "➕ Add New User", "🔍 Content-Based Suggestions"])

# ========== TAB 1: USER-BASED RECOMMENDATIONS ==========
with tab1:
    df, tools_df, purchase_matrix, sim_df, product_choices = get_updated_data()
    st.write("## 📌 User-Based Product Recommendations")

    if purchase_matrix.empty:
        st.warning("No user data found. Please add users first.")
    else:
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
                            row = tools_df[tools_df['title_clean'] == best_match].iloc[0]
                            st.markdown(f"### [{prod}]({row['title_url']})")
                            try:
                                st.image(row['image'], width=400)
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
    new_user_specialty = st.text_input("🔹 Medical Specialty (e.g., Orthopedic, Neuro):")

    if st.button("✅ Add User and Generate Recommendations"):
        if new_user_id.strip() == "" or new_user_purchases.strip() == "" or new_user_specialty.strip() == "":
            st.warning("Please enter all user details.")
        else:
            cursor.execute("SELECT COUNT(*) FROM users WHERE userID=?", (new_user_id,))
            if cursor.fetchone()[0] > 0:
                st.warning("User ID already exists. Please choose another one.")
            else:
                cursor.execute("INSERT INTO users (userID, previousPurchases, specialty) VALUES (?, ?, ?)",
                               (new_user_id.strip(), new_user_purchases.strip(), new_user_specialty.strip()))
                conn.commit()
                st.success(f"User '{new_user_id}' added successfully!")
                st.cache_data.clear()

# ========== TAB 3: CONTENT-BASED ==========
with tab3:
    df, tools_df, *_ = get_updated_data()
    st.write("## 🧠 Content-Based Product Filtering")

    if tools_df.empty:
        st.warning("No tool data available.")
    else:
        tool_titles = tools_df['title'].tolist()
        selected_tool = st.selectbox("🔎 Select a tool to find similar ones", tool_titles)

        if selected_tool:
            st.subheader("🔁 Similar Products")
            similar_df = similar_products_view(tools_df, selected_tool)
            for _, row in similar_df.iterrows():
                st.markdown(f"### [{row['title']}]({row['title_url']})")
                try:
                    st.image(row['image'], width=400)
                except:
                    st.write("(Image unavailable)")

        st.markdown("---")
        specialty_input = st.text_input("🩺 Enter a medical specialty to get recommended tools")

        if specialty_input:
            st.subheader(f"🔧 Tools for '{specialty_input}' Specialty")
            spec_df = products_by_specialty(tools_df, specialty_input)
            for _, row in spec_df.iterrows():
                st.markdown(f"### [{row['title']}]({row['title_url']})")
                try:
                    st.image(row['image'], width=400)
                except:
                    st.write("(Image unavailable)")
