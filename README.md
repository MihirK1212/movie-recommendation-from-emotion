# Emotion Detection and Review Analysis for Movie Recommendation
This is a project that recommends you a list of movies based on the review you have provided for a movie and the facial emotions you displayed while watching that movie.

## How to run
Clone the repo locally
```
git clone https://github.com/MihirK1212/movie-recommendation-from-emotion.git
```
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
Download weights
```
Downoload 'emotion_detector.h5' and 'weights.h5' from the drive link given below and place them in the project folder
https://drive.google.com/drive/folders/1dIUeL1p2Izh8kWTUq0YzlAYrx-4gw6P3?usp=sharing
```

Run the script
```
python movie_reccomendation.py
```
