import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


df=pd.read_csv("movie_dataset.csv")

features=['genres','keywords','director','overview','cast','vote_average']

for feature in features:
	df[feature]=df[feature].fillna('')   #Removes any NaN values so that combine_features does not give error

def combine_features(row):
	s=""
	for feature in features:
		s=s+str(row[feature])+" "
	s=s.strip()
	return s


df["combined_features"]=df.apply(combine_features,axis=1)

#Now df["combined_features"] is a table where each row=a movie and columns=features

#print(df["combined_features"].head())

count_vectorizer = CountVectorizer()
count_matrix=count_vectorizer.fit_transform(df["combined_features"])

similarity_scores=cosine_similarity(count_matrix)

movie_user_likes = "The Shawshank Redemption"

movie_index= get_index_from_title(movie_user_likes) #Returns index, i.e. 0 or 1 or 2 for our chosen movie
similar_movies=list(enumerate(similarity_scores[movie_index]))

#similar_movies is a list of the form [(0,similarity of 0 and i),(1,similarity of 1 and i),...(i,1),....]
# Where i=movie_index

sorted_similar_movies=sorted(similar_movies,key= lambda x:x[1],reverse=True) #Sort in reverse acc to second element

i=0
for movie in sorted_similar_movies:
	print(get_title_from_index(movie[0]))
	i=i+1
	if i>25:
		break
