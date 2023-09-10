# Emotion Detection and Review Analysis for movie recommendation
This is a project that recommends you a list of movies based on the review you have provided for a movie and the facial emotions you displayed while watching that movie.

## How to run
Clone the repo locally
```
git clone git@github.com:amitmakkad/G12_Healthcare.git !!!!!!!CHANGE THIS!!!!!!
```
Now go to repo directory
```
cd G12_Healthcare/ !!!!!!!CHANGE THIS!!!!!!
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
Downoload 'emotion_detector.h5' and 'weights.h5' from the drive link given below and place them in the project folder:

```

Run the script
```
python movie_reccomendation.py
```