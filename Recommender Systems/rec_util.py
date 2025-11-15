'''rec_sys.py
Utility/helper functions to help with analysis with recommender systems
Andrew Welling
CS 252: Mathematical Data Analysis and Visualization
Spring 2025
'''
import numpy as np
import matplotlib.pyplot as plt


def user_item_matrix(df, num_users, num_movies):
    '''Converts data where each sample is a single user rating to a user-item matrix

    Parameters:
    -----------
    df: pandas DataFrame. Each sample corresponds to the one user (`userId`) rating (`rating`) one movie (`movieId`).
    num_users: int. Number of unique users in the dataset (N).
    num_movies: int. Number of unique movies in the dataset (M).

    Returns:
    -----------
    user-item matrix ("A" matrix). ndarray. shape=(N, M). Matrix where rows correspond to users, columns correspond to
        movies. Unrated movies are represented with a 0 rating (the minimum actual rating given by users is 0.5).

    NOTE: It may be help to convert the columns of the DataFrame that you extract to NumPy ndarrays.
    '''
    return df.pivot(index='userId', columns='movieId', values='rating').fillna(0).to_numpy()


def get_top_rated_movies(A, movie_map, user_id, num_top_rated):
    '''Determines the `num_top_rated` movies that `user_id` rated most highly.

    Parameters:
    -----------
    A: ndarray. shape=(N, M). User-item matrix of star ratings ranging from 0. to 5.
            0 means movie is unrated by a user.
    movie_map: dictionary. keys=movieId (int), values=movie title (str).
    user_id: int. The ID of the user whose top rated movies we would like to determine.
    num_top_rated: int. Number of top rated movies that should be determined.

    Returns:
    -----------
    list. len=num_top_rated. String names of user `user_id`'s top rated movies.
        Format: [User's most highly rated movie, User's 2nd most highly rated movie, ...]
    '''
    user_ratings = A[user_id, :]
    top_indices = np.argsort(user_ratings)[::-1][:num_top_rated]  # get indices of top 10 movies
    return [movie_map[i] for i in top_indices]


def get_top_recommendations(A, rec_model, movie_map, user_id, num_top_recommendations):
    '''Determines the `num_top_recommendations` movie recommendations among the movies that user `user_id` has NOT rated.

    Parameters:
    -----------
    A: ndarray. shape=(N, M). User-item matrix of star ratings ranging from 0. to 5.
            0 means movie is unrated by a user.
    rec_model: Recommender. One of your fitted recommender systems.
    movie_map: dictionary. keys=movieId (int), values=movie title (str).
    user_id: int. The ID of the user for which we would like to recommend movies.
    num_top_recommendations: int. Number of top movie recommendations we would like to make.

    Returns:
    -----------
    list. len=num_top_recommendations. String names of top recommended movies for user `user_id`.
        Format: [Top recommendation for user, 2nd best recommendation for user, ...]
    ndarray. shape=(num_top_recommendations,). Floats representing the predicted numerical ratings of the top
        recommended movies.
    '''
    curr_user_ratings = A[user_id, :]
    unrated_ids = np.where(curr_user_ratings == 0)[0]
    predicted_ratings = rec_model.predict_user_item_ratings([user_id],unrated_ids)
    top_indices = np.argsort(predicted_ratings)[::-1][:num_top_recommendations]  # get indices of top 10 movies

    top_unrated_movie_ids = unrated_ids[top_indices]
    top_titles = [movie_map[i] for i in top_unrated_movie_ids]

    return top_titles, predicted_ratings[top_indices]



def plot_recommendations(user_id, top_movie_titles, top_movie_ratings, figsize=(5, 7)):
    '''Creates a side bar chart to visualize top recommended movies alongside annotations indicating the predicted
    rating.

    This function is provided. It should not need to be modified.

    Parameters:
    -----------
    user_id: int. The ID of the user for which we are recommending movies.
    top_movie_titles: list. Strings of the top recommended movies, ordered high->low.
    top_movie_ratings: list. Float predicted ratings of the top recommended movies, ordered high->low.
    figsize: tuple. Height and width of the bar chart.
    '''
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title("Top recommendation movies of user "+str(user_id))
    ax.invert_yaxis()

    bars = ax.barh(np.arange(len(top_movie_titles)), top_movie_ratings)
    ax.set_xlabel("Predicted ratings")
    ax.set_yticks(np.arange(len(top_movie_titles)), labels=top_movie_titles)

    for bar, rating in zip(bars, top_movie_ratings):
        ax.text(rating, bar.get_y() + bar.get_height() / 2, f'{rating:.2f}', va='center', ha='left')
