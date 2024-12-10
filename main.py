import pandas as pd

import mysql.connector

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import pickle

from flask import Flask, jsonify, request
from flask_cors import CORS

# from pyvi import ViTokenizer

app = Flask(__name__)
CORS(app)

@app.route('/greet', methods=['GET'])
def greet():
    return jsonify({"message": "Hello world"})

@app.route('/api/calculate_distance', methods=['GET'])
def calculate_distance():
    try:
        # Kết nối đến MySQL
        connection = mysql.connector.connect(
            host="localhost",        # Máy chủ MySQL (hoặc địa chỉ IP)
            user="root",             # Tên người dùng MySQL
            password="", # Mật khẩu của MySQL
            database="webphim_test"  # Tên cơ sở dữ liệu bạn muốn kết nối
            # database="phimmoi"  # Tên cơ sở dữ liệu bạn muốn kết nối
        )
        if connection.is_connected():
            print("Kết nối thành công đến MySQL!")

            # Tạo DataFrame từ kết quả truy vấn
            query = "SELECT * FROM movies;"  
            df_movies = pd.read_sql(query, con=connection)
            df_movies['genres'] = {}
            df_movies['genres'] = df_movies['genres'].apply(lambda x: [] if pd.isna(x) else x)

            query = "SELECT * FROM genres;"
            df_genres = pd.read_sql(query, con=connection)
            # print(df_genres.head(5))
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM genre_movie")
            rows = cursor.fetchall()

            for row in rows:
                index = df_movies[df_movies['id'] == row[2]].index
                value = df_genres[df_genres['id'] == row[1]].iloc[0]['title']
                if not index.empty: 
                    df_movies.loc[index[0], 'genres'].append(value)

    except mysql.connector.Error as err:
        print("Lỗi khi kết nối:", err)
        data = {'code': 500, 'message': 'Failed to connect database'}
        return jsonify(data)
    finally:
        if connection.is_connected():
            connection.close()
            print("Đã đóng kết nối.")

    def combineFeatures(row):
        # return ViTokenizer.tokenize(str(row['title']) + ' ' + str(row['original_name']) + ' ' + str(row['description']) + ' ' + ' '.join(row['genres']))
        return str(row['title']) + ' ' + str(row['original_name']) + ' ' + str(row['description']) + ' ' + ' '.join(row['genres'])


    df_movies['combine_features'] = df_movies.apply(combineFeatures, axis = 1)
    print(df_movies.iloc[0]['combine_features'])

    tf = TfidfVectorizer()

    tfMatrix = tf.fit_transform(df_movies['combine_features'])
    similarity = cosine_similarity(tfMatrix)
    print(similarity)

    pickle.dump(similarity, open('result/similarity.pkl', 'wb'))
    pickle.dump(df_movies, open('result/movie_list.pkl', 'wb'))
    data = {'code': 200, 'message': 'success'}
    return jsonify(data)

@app.route('/api/recommend', methods=['GET'])
def get_data():
    movie_id = request.args.get('id')
    number = 12 if request.args.get('number') == None else int(request.args.get('number'))

    user_id = request.args.get('user_id')
    df_watched_movies = None
    if user_id is not None:
        try:
            # Kết nối đến MySQL
            connection = mysql.connector.connect(
                host="localhost",        # Máy chủ MySQL (hoặc địa chỉ IP)
                user="root",             # Tên người dùng MySQL
                password="", # Mật khẩu của MySQL
                database="webphim_test"  # Tên cơ sở dữ liệu bạn muốn kết nối
                # database="phimmoi"  # Tên cơ sở dữ liệu bạn muốn kết nối
            )
            if connection.is_connected():
                print("Kết nối thành công đến MySQL!")
                query = "SELECT movie_id FROM histories WHERE user_id = %s AND updated_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH);"  
                df_watched_movies = pd.read_sql(query, con=connection, params=(user_id,))
                print(df_watched_movies['movie_id'])

        except mysql.connector.Error as err:
            print("Lỗi khi kết nối:", err)
            data = {'code': 500, 'message': 'Failed to connect database'}
            return jsonify(data)
        finally:
            if connection.is_connected():
                connection.close()
                print("Đã đóng kết nối.")

    movie_id = int(movie_id)
    df_movies = pd.read_pickle('result/movie_list.pkl')
    similarity = pickle.load(open('result/similarity.pkl', 'rb'))

    if movie_id not in df_movies['id'].values:
        return jsonify({'Error': 'Id không hợp lệ'})
    
    index_find = df_movies[df_movies['id'] == movie_id].index[0]
    similarityMovies = list(enumerate(similarity[index_find]))

    sortedSimilarityMovies = sorted(similarityMovies, key = lambda x:x[1], reverse=True)

    def getName(index):
        return (df_movies[df_movies.index == index]['title'].values[0])
    def getId(index):
        return (df_movies[df_movies.index == index]['id'].values[0])

    result = []
    if df_watched_movies is not None:
        cnt = 0
        for i in range(1, len(sortedSimilarityMovies)):
            if not int(getId(sortedSimilarityMovies[i][0])) in df_watched_movies['movie_id'].values:
                print(getName(sortedSimilarityMovies[i][0]) + ' ' + str(sortedSimilarityMovies[i][1]))
                result.append({
                    'id': str(getId(sortedSimilarityMovies[i][0])),
                    'name': getName(sortedSimilarityMovies[i][0]),
                })
                cnt += 1
            if cnt == number:
                break
    else:
        for i in range(1, number + 1):
            print(getName(sortedSimilarityMovies[i][0]) + ' ' + str(sortedSimilarityMovies[i][1]))
            result.append({
                'id': str(getId(sortedSimilarityMovies[i][0])),
                'name': getName(sortedSimilarityMovies[i][0]),
            })
    data = {'movie': getName(index_find), 'movie_recommender': result}
    return jsonify(data)

@app.route('/api/recommend/history', methods=['GET'])
def get_data_base_history():
    number = 12 if request.args.get('number') == None else int(request.args.get('number'))

    user_id = request.args.get('user_id')
    df_watched_movies = None
    if user_id is not None:
        try:
            # Kết nối đến MySQL
            connection = mysql.connector.connect(
                host="localhost",        # Máy chủ MySQL (hoặc địa chỉ IP)
                user="root",             # Tên người dùng MySQL
                password="", # Mật khẩu của MySQL
                database="webphim_test"  # Tên cơ sở dữ liệu bạn muốn kết nối
                # database="phimmoi"  # Tên cơ sở dữ liệu bạn muốn kết nối
            )
            if connection.is_connected():
                print("Kết nối thành công đến MySQL!")
                query = "SELECT movie_id FROM histories WHERE user_id = %s AND updated_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH) ORDER BY updated_at DESC;"  
                df_watched_movies = pd.read_sql(query, con=connection, params=(user_id,))
                df_watched_movies_base = df_watched_movies[0:5]
                print(df_watched_movies_base['movie_id'])

        except mysql.connector.Error as err:
            print("Lỗi khi kết nối:", err)
            data = {'code': 500, 'message': 'Failed to connect database'}
            return jsonify(data)
        finally:
            if connection.is_connected():
                connection.close()
                print("Đã đóng kết nối.")

    df_movies = pd.read_pickle('result/movie_list.pkl')
    similarity = pickle.load(open('result/similarity.pkl', 'rb'))

    watched_movies_index = []

    for movie_id in df_watched_movies_base['movie_id'].values:
        if movie_id in df_movies['id'].values:
            index_find = df_movies[df_movies['id'] == movie_id].index[0]
            watched_movies_index.append(index_find)

    print(watched_movies_index)
    if len(watched_movies_index) == 0:
        data = {'movie_recommender': []}
        return jsonify(data)
    average_similarity = np.mean([similarity[idx] for idx in watched_movies_index], axis=0)
    sortedSimilarityMovies = sorted(list(enumerate(average_similarity)), key = lambda x:x[1], reverse=True)

    # if movie_id not in df_movies['id'].values:
    #     return jsonify({'Error': 'Id không hợp lệ'})
    
    # index_find = df_movies[df_movies['id'] == movie_id].index[0]
    # similarityMovies = list(enumerate(similarity[index_find]))

    # sortedSimilarityMovies = sorted(similarityMovies, key = lambda x:x[1], reverse=True)

    def getName(index):
        return (df_movies[df_movies.index == index]['title'].values[0])
    def getId(index):
        return (df_movies[df_movies.index == index]['id'].values[0])

    result = []
    if df_watched_movies is not None:
        cnt = 0
        for i in range(1, len(sortedSimilarityMovies)):
            if not int(getId(sortedSimilarityMovies[i][0])) in df_watched_movies['movie_id'].values:
                print(getName(sortedSimilarityMovies[i][0]) + ' ' + str(sortedSimilarityMovies[i][1]))
                result.append({
                    'id': str(getId(sortedSimilarityMovies[i][0])),
                    'name': getName(sortedSimilarityMovies[i][0]),
                })
                cnt += 1
            if cnt == number:
                break
    else:
        for i in range(1, number + 1):
            print(getName(sortedSimilarityMovies[i][0]) + ' ' + str(sortedSimilarityMovies[i][1]))
            result.append({
                'id': str(getId(sortedSimilarityMovies[i][0])),
                'name': getName(sortedSimilarityMovies[i][0]),
            })
    data = {'movie_recommender': result}
    return jsonify(data)


if __name__ == '__main__':
    app.run(port=5555)