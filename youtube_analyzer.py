import re
import pandas as pd
from googleapiclient.discovery import build

# Define your API key here
API_KEY = 'AIzaSyAa2lnBRCPoVbmdjo6ZoPo4YM-PGca1ewE'

# Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# This function now accepts 'model' as input!
def analyze_youtube_comments(video_url, model):
    # Extract video ID from URL
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if video_id_match:
        video_id = video_id_match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

    # Fetch comments
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    # Predict toxicity
    results = []
    for comment in comments:
        pred, confidence = CheckToxic(comment, model)
        label = 'Toxic' if pred == 1 else 'Non-Toxic'
        results.append({
            'Comment': comment,
            'Prediction': label,
            'Confidence': round(confidence * 100, 2)
        })

    results_df = pd.DataFrame(results)
    return results_df

# Helper function to predict (takes text and model)
def CheckToxic(text, model):
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    return pred, max(proba)


