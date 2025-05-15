import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

# ------------------------- Page Setup -------------------------
st.set_page_config(page_title="🍽️ Recipe Recommender", layout="wide")
st.title("🍽️ **Recipe Recommender**")

# Custom Styling with Pink color accents and Custom Fonts (Roboto)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        
        body {
            background-color: #F8F0F5;
            font-family: 'Roboto', sans-serif;
            color: #333;
        }
        h1 {
            color: #D14D74;  /* Lighter shade of pink */
            font-weight: 700;
        }
        h2, h3 {
            color: #D14D74;
            font-weight: 500;
        }
        .stButton>button {
            background-color: #D14D74; /* Button with light pink color */
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #E7A1B9; /* Slightly lighter shade on hover */
        }
        .stTextInput>div>input {
            border-radius: 10px;
            border: 2px solid #D14D74;
            padding: 10px;
            font-family: 'Roboto', sans-serif;
        }
        .stMarkdown {
            font-size: 18px;
            font-family: 'Roboto', sans-serif;
        }
        .stWarning {
            background-color: #F8D0D9;
            color: #D14D74;
        }
        .stImage {
            border-radius: 8px;
        }
        
        /* Custom Welcome Message */
        .welcome-message {
            font-family: 'Roboto', sans-serif;
            font-size: 24px;
            color: #D14D74;
            background-color: #F8F0F5;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 40px;
        }
        .welcome-message h1 {
            font-size: 36px;
            font-weight: 700;
            color: #D14D74;
        }
        .welcome-message p {
            font-size: 18px;
            line-height: 1.6;
            color: #555;
        }
        .welcome-message ul {
            font-size: 18px;
            color: #555;
            text-align: left;
            margin-left: 20px;
        }
        .welcome-message li {
            margin-bottom: 15px;
        }
        .button-container {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------- Load & Clean Datasets -------------------------
# Read CSVs
df1 = pd.read_csv("Food_Recipe.csv")
df2 = pd.read_csv("IndianHealthyRecipe.csv")
df3 = pd.read_csv("Recipes from around the world.csv", encoding='latin1')
df4 = pd.read_csv("Indonesian_Food_Recipes.csv")  # New dataset added

# Standardize column names
for df in [df1, df2, df3, df4]:
    df.columns = df.columns.str.lower().str.strip()

# Ensure essential columns exist
required_columns = ['name', 'ingredients_name', 'ingredients_quantity', 'image_url']
for df in [df1, df2, df3, df4]:
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

# Concatenate all into one DataFrame
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# ------------------------- Ingredient Preparation -------------------------
df['ingredients'] = (df['ingredients_name'].fillna('') + ' ' + df['ingredients_quantity'].fillna('')).astype(str)

def clean_ingredient_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s,]', '', text)
    return text

df['ingredients_joined'] = df['ingredients'].apply(clean_ingredient_text)
df = df[df['ingredients_joined'].str.strip() != ""]

# ------------------------- Vectorization -------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients_joined'])

# ------------------------- Icons -------------------------
icons_directory = 'icons/'
ingredient_icons = {
    "turmeric": "turmeric.png",
    "cumin": "cumin.png",
    "coriander": "coriander.png",
    "cardamom": "cardamom.png",
    "cloves": "cloves.png",
    "mustard": "mustard.png",
    "ginger": "ginger.png",
    "garlic": "garlic.png",
    "chili": "chili.png",
    "asafoetida": "asafoetida.png",
    "curry leaves": "curry_leaves.png",
    "fenugreek": "fenugreek.png",
    "tamarind": "tamarind.png",
    "coconut": "coconut.png",
    "saffron": "saffron.png",
    "bay leaf": "bay_leaf.png",
    "black pepper": "black_pepper.png",
    "ghee": "ghee.png",
    "yogurt": "yogurt.png",
    "paneer": "paneer.png",
    "lentils": "lentils.png",
    "dal": "dal.png",
    "rice": "rice.png",
    "wheat": "wheat.png",
    "chapati": "chapati.png",
    "naan": "naan.png",
    "roti": "roti.png",
    "spinach": "spinach.png",
    "tomato": "tomato.png",
    "onion": "onion.png",
    "potato": "potato.png",
    "carrot": "carrot.png",
    "cauliflower": "cauliflower.png",
    "peas": "peas.png",
    "pumpkin": "pumpkin.png",
    "brinjal": "brinjal.png",
    "bottle gourd": "bottle_gourd.png",
    "drumstick": "drumstick.png",
    "okra": "okra.png",
    "sweet potato": "sweet_potato.png",
    "green beans": "green_beans.png",
    "pumpkin seeds": "pumpkin_seeds.png",
    "pomegranate": "pomegranate.png",
    "banana": "banana.png",
    "apple": "apple.png",
    "papaya": "papaya.png",
    "mango": "mango.png",
    "lemon": "lemon.png",
    "lime": "lime.png"
}

def get_ingredient_icon(ingredient):
    for key, value in ingredient_icons.items():
        if key in ingredient.lower():
            return os.path.join(icons_directory, value)
    return None

# ------------------------- Input -------------------------
user_input_ingredients = st.text_input("🔍 Enter ingredients (e.g., tomato, onion, garlic)")

# ------------------------- Recommendation Function -------------------------
def recommend_recipes(user_ingredients, top_n=10):
    user_input = clean_ingredient_text(', '.join(user_ingredients))
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    if all(similarity_scores[i] < 0.1 for i in top_indices):
        st.warning("⚠️ No relevant recipes found. Try adding more common ingredients.")
        return

    st.markdown("### Recommended Recipes 🍽️")
    for i in top_indices:
        recipe_name = df.iloc[i]['name']
        recipe_image_url = df.iloc[i]['image_url']
        ingredients_text = df.iloc[i]['ingredients']

        ingredients_with_icons = []
        for ingredient in ingredients_text.split(','):
            icon_path = get_ingredient_icon(ingredient.strip())
            if icon_path and os.path.exists(icon_path):
                ingredients_with_icons.append(f"![{ingredient}]({icon_path}) {ingredient}")
            else:
                ingredients_with_icons.append(ingredient)
        ingredients_with_icons = ' | '.join(ingredients_with_icons)

        st.markdown(f"#### {recipe_name}")
        st.markdown(f"**Ingredients**: {ingredients_with_icons}")
        st.markdown(f"**Similarity Score**: {similarity_scores[i]:.2f}")
        if recipe_image_url:
             st.image(recipe_image_url, use_container_width=True)
        st.markdown("---")

# ------------------------- Welcome Message ------------------------`-
st.markdown("""
    <div class="welcome-message">
        <h1>Welcome to the Recipe Recommender 🍽️!</h1>
        <p>Craving something delicious? We've got you covered! 🎉</p>
        <p>Simply enter the ingredients you have, and let us help you discover exciting recipes you can make with what you have at home!</p>
        <p>From Indian dishes to international cuisines, explore a variety of recipes tailored just for you. 🌍🍴</p>
        
        
<h3 style="font-size: 24px; font-weight: 600; color: #D14D74;">How it works:</h3>
<ul>
    <li>Enter a few ingredients (e.g., tomato, onion, garlic) in the input box below.</li>
    <li>Click <strong>Recommend</strong> to discover recipes you can make with those ingredients.</li>
    <li>Start cooking and enjoy your creation! 👨‍🍳👩‍🍳</li>
</ul>

<p>Ready to cook something delicious? Let’s get started! 🍲✨</p>
    </div>
""", unsafe_allow_html=True)

# ------------------------- Trigger Button -------------------------
if st.button("🔎 Recommend"):
    if user_input_ingredients:
        user_input_list = [x.strip() for x in user_input_ingredients.split(',')]
        recommend_recipes(user_input_list)
    else:
        st.warning("⚠️ Please enter some ingredients.")

# ------------------------- Footer -------------------------
st.markdown("---")
st.markdown("Created with using Streamlit")
