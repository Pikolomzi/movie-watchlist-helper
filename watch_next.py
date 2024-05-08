import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_movie_descriptions(file_path):
    """
    Load movie descriptions from a file.

    Args:
        file_path (str): Path to the file containing movie descriptions.

    Returns:
        list: List of movie descriptions.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except UnicodeDecodeError:
        print(f"Error: Unable to decode file '{file_path}'. Please ensure it's encoded properly.")
        return []

def find_similar_movie(query_description, movie_descriptions):
    """
    Find the index of the most similar movie description to the given query description.

    Args:
        query_description (str): Description of the movie to find similarity for.
        movie_descriptions (list): List of movie descriptions.

    Returns:
        int: Index of the most similar movie description.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movie_descriptions + [query_description])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    similar_movie_index = np.argmax(similarities)
    return similar_movie_index

def main():
    """
    Main function to find similar movies based on a query description.
    """
    movie_descriptions = load_movie_descriptions('movies.txt')
    if not movie_descriptions:
        print("No movie descriptions loaded. Exiting.")
        return

    query_description = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator."
    similar_movie_index = find_similar_movie(query_description, movie_descriptions)
    print("You may want to watch:", movie_descriptions[similar_movie_index])

if __name__ == "__main__":
    main()
