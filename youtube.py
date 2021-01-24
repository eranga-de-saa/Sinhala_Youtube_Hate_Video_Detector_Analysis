__author__ = "Y-Nots"

import os
import urllib.request
import pandas as pd
from urllib.parse import urlparse, urlencode, parse_qs
from youtube_data import VideoData


def main():
    keys = pd.read_csv(os.getcwd() + "\\keys.csv", encoding='utf-')
    urls = pd.read_csv(os.getcwd() + "\\videoIDs.csv", encoding='utf-')
    noOfKeys = keys.shape[0]
    noOfUrls = urls.shape[0]
    print(noOfKeys)
    print(noOfUrls)

    for x in range(noOfUrls):
        url = urls.iloc[x, 0]
        status = urls.iloc[x, 1]
        category=urls.iloc[x, 2]
        sentiment = urls.iloc[x, 3]
        if status == 1:
            print(str(x) + ":AlreadyDone")
        else:
            video_id = urlparse(url)
            q = parse_qs(video_id.query)
            vid = q["v"][0]
            key = keys.iloc[x % noOfKeys, 0]
            # print(key)
            # AIzaSyDKJRnYQxDlzR23YEWTx0t7GgRuY4srAJQ
            vc = VideoData(vid, key, x, category, sentiment)
            vc.get_video_comments()
            urllib.request.urlretrieve("https://img.youtube.com/vi/" + vid + "/hqdefault.jpg",
                                       "output/" + str(x) + ".jpg")
            print(str(x) + ":Done")
            urls.iloc[x, 1] = '1'
            urls.to_csv("videoIDs.csv", index=False, encoding='utf-8')



if __name__ == '__main__':
    main()
