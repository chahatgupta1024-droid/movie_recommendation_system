import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("movies.csv")

# Convert genres into vectors
cv = CountVectorizer()
count_matrix = cv.fit_transform(data["genres"])

# Calculate cosine similarity
similarity = cosine_similarity(count_matrix)

# Recommendation function
def recommend(movie_name):
    movie_index = data[data["title"] == movie_name].index[0]
    similar_movies = list(enumerate(similarity[movie_index]))
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:")
    for i in sorted_movies[1:6]:
        print(data.iloc[i[0]]["title"])

# User input
movie = input("Enter movie name: ")
recommend(movie)