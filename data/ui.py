import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Data loading
@st.cache_data
def load_data():
    # Use absolute paths for the CSV files
    data_dir = r"c:\Users\ASHISH\Projects\MovieLens Insight System\data"
    movies_df = pd.read_csv(f"{data_dir}/Movies.csv", encoding='latin1')
    ratings_df = pd.read_csv(f"{data_dir}/Ratings.csv", encoding='latin1')
    users_df = pd.read_csv(f"{data_dir}/Users.csv", encoding='latin1')

    movies_df['Year'] = movies_df['Title'].str.extract(r'\((\d{4})\)')
    movies_df['Year'] = pd.to_numeric(movies_df['Year'], errors='coerce').astype('Int64')
    movies_df = movies_df.dropna(subset=['Year'])
    movies_df = movies_df[movies_df['Year'] <= 2025]
    movies_df['Category'] = movies_df['Category'].str.split('|')
    movies_df = movies_df.explode('Category')

    merged = pd.merge(ratings_df, movies_df, on='MovieID')
    merged = pd.merge(merged, users_df, on='UserID')
    return movies_df, ratings_df, users_df, merged

movies_df, ratings_df, users_df, merged = load_data()

age_map = {
    1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49",
    50: "50-55", 56: "Above 56"
}
occupation_map = {
    0: "Not specified", 1: "Academician", 2: "Artist", 3: "Admin/Office Work",
    4: "Grad Student", 5: "Customer Service", 6: "Doctor", 7: "Executive",
    8: "Farmer", 9: "Homemaker", 10: "K-12 Student", 11: "Lawyer", 12: "Programmer",
    13: "Retired", 14: "Sales & Marketing", 15: "Scientist", 16: "Self-Employed",
    17: "Engineer", 18: "Tradesman", 19: "Unemployed", 20: "Writer"
}

# Sidebar
st.sidebar.title("🎬 Movie Analytics System")
analysis_type = st.sidebar.radio("Select Analysis", [
    "Movies per Year",
    "Top Category per Year",
    "Popular Category by Age",
    "Cluster Age vs Category",
    "Year-wise Count",
    "Year & Category Count",
    "Cluster Category vs Occupation",
    "Predict Category (ML)",
    "Category Demographics"
])

st.sidebar.markdown("#### ℹ️ Age & Occupation Reference")
st.sidebar.text("\n".join([f"{k}: {v}" for k, v in age_map.items()]))
st.sidebar.markdown("---")
st.sidebar.text("\n".join([f"{k}: {v}" for k, v in occupation_map.items()]))

# Analysis Pages
if analysis_type == "Movies per Year":
    movie_year_df = movies_df.groupby('Year')['MovieID'].count().reset_index(name='MovieCount')
    st.line_chart(movie_year_df.rename(columns={"Year": "index"}).set_index("index"))

elif analysis_type == "Top Category per Year":
    top_cat = merged.groupby(['Year', 'Category'])['Rating'].mean().reset_index()
    top = top_cat.sort_values(['Year', 'Rating'], ascending=[True, False]) \
                 .drop_duplicates('Year') \
                 .sort_values('Year')
    st.dataframe(top)

elif analysis_type == "Popular Category by Age":
    merged['AgeGroup'] = pd.cut(
        merged['Age'],
        bins=[0, 18, 25, 35, 45, 56, 100],
        labels=['<18', '18-25', '26-35', '36-45', '46-56', '56+'],
        right=False
    )
    user_counts = merged.groupby(['AgeGroup', 'Category'], observed=True)['UserID'].nunique().reset_index(name='UserCount')
    max_indices = user_counts.groupby('AgeGroup', observed=True)['UserCount'].idxmax()
    result = user_counts.loc[max_indices].sort_values(by='AgeGroup')
    st.dataframe(result)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(result['AgeGroup'].astype(str), result['UserCount'], color='skyblue')
    for bar, category in zip(bars, result['Category']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, category, ha='center', rotation=45)
    st.pyplot(fig)

elif analysis_type == "Cluster Age vs Category":
    data = merged.groupby(['Age', 'Category'])['Rating'].count().unstack(fill_value=0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(scaled_data)
    fig, ax = plt.subplots()
    for cluster in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster]
        ax.scatter(cluster_data.index, [cluster]*len(cluster_data), label=f"Cluster {cluster}")
    ax.set_title("KMeans Clustering (Age vs Category)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Cluster")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(data[['Cluster']])

elif analysis_type == "Year-wise Count":
    st.write(movies_df.groupby('Year')['MovieID'].count())

elif analysis_type == "Year & Category Count":
    out = movies_df.groupby(['Year', 'Category'])['MovieID'].count().reset_index()
    st.dataframe(out)

elif analysis_type == "Cluster Category vs Occupation":
    pivot = merged.groupby(['Occupation', 'Category'])['Rating'].count().reset_index()
    pivot_table = pivot.pivot(index='Occupation', columns='Category', values='Rating').fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot_table)
    kmeans = KMeans(n_clusters=3, random_state=42)
    pivot_table['Cluster'] = kmeans.fit_predict(X_scaled)

    occ_code = st.number_input("Enter Occupation Code", min_value=0, max_value=20, step=1)
    if occ_code in pivot_table.index:
        cluster = pivot_table.loc[occ_code, 'Cluster']
        top_cats = pivot_table.loc[occ_code].drop('Cluster').sort_values(ascending=False).head(3)
        st.write(f"Cluster: {cluster}")
        st.write("Top Categories:")
        st.write(top_cats[top_cats > 0])

elif analysis_type == "Predict Category (ML)":
    try:
        model = joblib.load('rf_movie_category_model.pkl')
        le = joblib.load('category_label_encoder.pkl')
    except Exception as e:
        st.error("Model or encoder not found.")
        st.stop()

    occ = st.number_input("Occupation Code", min_value=0, max_value=20, step=1)
    age = st.number_input("Age", min_value=1, max_value=100, step=1)

    if st.button("Predict"):
        try:
            pred = model.predict([[occ, age]])[0]
            category = le.inverse_transform([pred])[0]
            st.success(f"Predicted Category: {category}")
        except Exception as e:
            st.error("Prediction failed.")

elif analysis_type == "Category Demographics":
    categories = sorted(merged['Category'].unique())
    cat = st.selectbox("Choose a Category", categories)
    cat_data = merged[merged['Category'] == cat]
    if not cat_data.empty:
        most_likely_age = cat_data['Age'].mode().iloc[0]
        most_likely_occ = cat_data['Occupation'].mode().iloc[0]
        avg_age = cat_data['Age'].mean()
        total_viewers = len(cat_data)

        st.markdown(f"""
        ### 📊 {cat}
        - 👤 Most Likely Age: **{age_map.get(most_likely_age, str(most_likely_age))}**
        - 💼 Most Likely Job: **{occupation_map.get(most_likely_occ, 'Unknown')}**
        - 📈 Average Age: **{avg_age:.1f}**
        - 👥 Total Viewers: **{total_viewers:,}**
        """)
    else:
        st.warning("No data for selected category.")
