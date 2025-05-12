import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load train.json
with open('C:/Users/KIIT/recipe_data/train.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Handle missing ingredients
df['ingredients'] = df['ingredients'].fillna([])  # Handle missing ingredients
df['ingredients_joined'] = df['ingredients'].apply(lambda x: ', '.join(x))

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients_joined'])

# Recommendation function
def recommend_recipes(user_ingredients):
    # Join the user input ingredients into a string
    user_input = ', '.join(user_ingredients)
    user_vector = vectorizer.transform([user_input])
    
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Get top 5 recipes
    top_indices = similarity_scores.argsort()[::-1][:5]
    recommendations = []
    
    for i in top_indices:
        recommendations.append({
            'Recipe ID': df.iloc[i]['id'],
            'Cuisine': df.iloc[i]['cuisine'],
            'Ingredients': df.iloc[i]['ingredients'],
            'Similarity Score': round(similarity_scores[i] * 100, 2)
        })
        
    return recommendations

# Example usage
user_input_ingredients = ["tomato", "garlic", "onion"]
recommended_recipes = recommend_recipes(user_input_ingredients)

# Print recommendations
for recipe in recommended_recipes:
    print(f"Recipe ID: {recipe['Recipe ID']} | Cuisine: {recipe['Cuisine']}")
    print(f"Ingredients: {', '.join(recipe['Ingredients'])}")
    print(f"Similarity Score: {recipe['Similarity Score']}%")
    print("-" * 50)
