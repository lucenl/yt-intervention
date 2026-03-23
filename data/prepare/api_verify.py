import pandas as pd
import os
from googleapiclient.discovery import build
from tqdm.auto import tqdm

def youtube_api_request(api_key, video_ids):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.videos().list(
        part="id",
        id=','.join(video_ids)
    )
    response = request.execute()
    return response.get('items', [])

def read_set(file):
    if not os.path.exists(file):
        return set()
    with open(file) as f:
        return set(f.read().split('\n'))

def write_set(file, setw):
    with open(file, 'w') as f:
        f.write('\n'.join(setw))

if __name__ == '__main__':

    # build video list
    # df = pd.read_pickle('lucen/watch_histories.pickle')
    df = pd.read_csv('../training/harmful.csv')
    videos = df['videoId'].tolist()

    # load cached data
    deleted_videos = read_set('lucen/deleted_videos.txt')
    existing_videos = read_set('lucen/existing_videos.txt')

    # enter youtube api key here
    api_key = "AIzaSyCGDOfQcocrRlzIC1ixUt60iluuEC3wBMk"

    # run in batches of 50
    for i in tqdm(range(0, len(videos), 50)):
        # make sure videos in this batch aren't already processed
        batch = [i for i in videos[i:i+50] if i and i not in deleted_videos and i not in existing_videos]

        # check if batch has some videos
        if batch:
            # send req to youtube api
            response = youtube_api_request(api_key, batch)

            # check videos included in response, deleted videos won't be in the response
            response_videos = [i['id'] for i in response]
            deleted_videos |= set([i for i in batch if i not in response_videos])
            existing_videos |= set([i for i in batch if i in response_videos])

    # save information
    write_set('lucen/deleted_videos.txt', deleted_videos)
    write_set('lucen/existing_videos.txt', existing_videos)