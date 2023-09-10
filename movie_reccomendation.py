import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras
import cv2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from os import system
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_hub as hub

model_emotion=keras.models.load_model("emotion_detector.h5")

df=pd.read_csv("movie_dataset.csv")

def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

def print_similiar_movies(movie_user_likes,movie_genre):

    if not ((df['title'] == movie_user_likes).any()):
        response_add=input("Movie name not valid...would you like to add this movie to our dataset? (Y/N) ")
        response_add.upper()
        if response_add=='Y':
            print("To skip a response, press the Enter key")
            index= df.shape[0] + 1
            genre=movie_genre
            keywords=input("Describe the Movie: ")
            language= input("Enter the languafe of the move (hi/en): ")
        else:
            return
    
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

    count_vectorizer = CountVectorizer()
    count_matrix=count_vectorizer.fit_transform(df["combined_features"])
    similarity_scores=cosine_similarity(count_matrix)
    movie_index= get_index_from_title(movie_user_likes) #Returns index, i.e. 0 or 1 or 2 for our chosen movie
    similar_movies=list(enumerate(similarity_scores[movie_index]))

    #similar_movies is a list of the form [(0,similarity of 0 and i),(1,similarity of 1 and i),...(i,1),....]
    # Where i=movie_index

    sorted_similar_movies=sorted(similar_movies,key= lambda x:x[1],reverse=True) #Sort in reverse acc to second element

    i=0
    for movie in sorted_similar_movies:
        print(get_title_from_index(movie[0]))
        i=i+1
        if i>10:
            break

def face_emotion(movie_genre):
    Classes=["Happy","Negative","Shocked"]
    emotion_detected=-1
    emotion_probability=0
    size = 4
    emotion_probability=-1000
    webcam = cv2.VideoCapture(0) #Use camera 0
    time.sleep(2)
    
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml') #Load the xml file

    i=10
    while True:
        (rval, im) = webcam.read()
        im=cv2.flip(im,1,1) #Obtain the flipped (mirror image)
        mini = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4)) #Reduce image size

        faces = classifier.detectMultiScale(mini) #Detect faces

        time_to_quit=-1
        #Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Increase the image size back to original
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(96,96))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,96,96,3))
            reshaped = np.vstack([reshaped])
            result=model_emotion.predict(reshaped)

            label=np.argmax(result,axis=1)[0]

            prob_str=str((result[0][label])*100)
            prob_value=(result[0][label])*100

            class_label=Classes[label]

            cv2.rectangle(im,(x,y-40),(x+w,y),(0,0,0),-1)
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,0),2)
            cv2.putText(im,class_label, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            if prob_value>60 and i>10:
                emotion_detected=label
                emotion_probability=prob_value
                time_to_quit=1
        
        i=i+1    

        cv2.imshow('LIVE',   im)
        key = cv2.waitKey(10)
        # if Esc key is press then break out of the loop
        if key == 27 or time_to_quit==1:
            time.sleep(2)
            break

    webcam.release()
    cv2.destroyAllWindows()
    cls = lambda: system('cls')
    cls()

    # print(emotion_detected,movie_genre)

    if emotion_detected>=0 and emotion_probability>60:
        if emotion_detected==0:
            response_yn1=input("Great! Seems like you liked the movie! (Y/N) ")
            response_yn1.upper()
            if response_yn1=='N':
                response_yn2=input("Sorry to hear that! Would you like to try again? (Y/N) ")
                response_yn2.upper()
                if response_yn2=='Y':
                    return face_emotion(movie_genre.lower())
                else:
                    return -1000
        elif emotion_detected==1 and movie_genre=='tragedy':
            response_yn1=input("Great! Seems like you liked the movie! (Y/N) ")
            response_yn1.upper()
            if response_yn1=='N':
                response_yn2=input("Sorry to hear that! Would you like to try again? (Y/N) ")
                response_yn2.upper()
                if response_yn2=='Y':
                    return face_emotion(movie_genre.lower())
                else:
                    return -1000
        elif emotion_detected==2 and movie_genre=='horror':
            response_yn1=input("Great! Seems like you liked the movie! (Y/N) ")
            response_yn1.upper()
            if response_yn1=='N':
                response_yn2=input("Sorry to hear that! Would you like to try again? (Y/N) ")
                response_yn2.upper()
                if response_yn2=='Y':
                    return face_emotion(movie_genre.lower())
                else:
                    return -1000
        elif emotion_detected==2 and movie_genre=='suspense':
            response_yn1=input("Great! Seems like you liked the movie! (Y/N) ")
            response_yn1.upper()
            if response_yn1=='N':
                response_yn2=input("Sorry to hear that! Would you like to try again? (Y/N) ")
                response_yn2.upper()
                if response_yn2=='Y':
                    return face_emotion(movie_genre.lower())
                else:
                    return -1000
        elif emotion_detected==2 and movie_genre=='action':
            response_yn1=input("Great! Seems like you liked the movie! (Y/N) ")
            response_yn1.upper()
            if response_yn1=='N':
                response_yn2=input("Sorry to hear that! Would you like to try again? (Y/N) ")
                response_yn2.upper()
                if response_yn2=='Y':
                    return face_emotion(movie_genre.lower())
                else:
                    return -1000
        else:
            response_yn2=input("Oh you didnt like the movie! Would you like to try again? (Y/N) ")
            response_yn2.upper()
            if response_yn2=='Y':
                return face_emotion(movie_genre.lower())
            else:
                return -1000
    else:
        response_yn=input("Sorry we couldn't detect any particular emotion on your face. Would you like to try agin? (Y/N) ")
        response_yn.upper()
        if response_yn=='Y':
            return face_emotion(movie_genre.lower())
        else:
            return 40

    return emotion_probability

def review_predictor(movie_review):
    train_examples_batch= tf.constant([b'Gray' b'wolf' b'Dog'])
    embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
    hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
    hub_layer(train_examples_batch[:3])
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    # model.summary()
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.load_weights('weights.h5')
    cls = lambda: system('cls')
    cls()
    review_prob= model.predict([movie_review])
    if review_prob>0 and review_prob<1:
        return 75
    if review_prob>1:
        return 100
    if review_prob<0 and review_prob>-1 :
        return 25
    else:
        return 0
    

def movie_recommendor():
    movie_name=input("Enter the name of the movie ")
    movie_genre=input("Enter the genre of the movie (Action/Comedy/Thriller/Tragedy/Horror) ")
    movie_review=input("Enter a review for the movie ")

    weighted_probability= face_emotion(movie_genre.lower()) + review_predictor(movie_review.lower())
    if weighted_probability>50:
        print_similiar_movies(movie_name,movie_genre)
    else:
        response_yn=input("Seems like you didnt like the movie! Would you like to try again?(Y/N)")
        response_yn.upper()
        if response_yn=="Y":
            return movie_recommendor()
        else:
            return


cls = lambda: system('cls')
cls()
movie_recommendor()

print("End")



