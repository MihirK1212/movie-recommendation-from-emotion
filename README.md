# Emotion Detection and Review Analysis for Movie Recommendation
This is a project that recommends you a list of movies based on the review you have provided for a movie and the facial emotions you displayed while watching that movie.

## Clone the Repistory
Clone the repo locally
```
git clone https://github.com/MihirK1212/movie-recommendation-from-emotion.git
```

## Download weights 
Downoload 'emotion_detector.h5' and 'weights.h5' from the drive [link](https://drive.google.com/drive/folders/1dIUeL1p2Izh8kWTUq0YzlAYrx-4gw6P3?usp=sharing) and place them in the project folder

## How to run
Now go to repo directory
```
cd movie-recommendation-from-emotion
```
Create a Virtual environment and activate it (for linux)
```
python -m venv myenv 
virtualenv myenv   
source myenv/bin/activate
```
Install dependencies
```
pip install -r requirements.txt
```
Run the script
```
python movie_reccomendation.py
```

