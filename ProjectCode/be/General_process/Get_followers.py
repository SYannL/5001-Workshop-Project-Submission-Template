import requests
import os
import json

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = os.environ.get("BEARER_TOKEN")
bearer_token='AAAAAAAAAAAAAAAAAAAAAP%2BjqgEAAAAAqrvFGhBVG%2F8SGH5x9SZP4bZzpm8%3DmduFHPia7yGjCHId38f9ZOZvQ9rW2NBfJ2SLRhZJfqaGBaK4hW'

import tweepy

consumer_key = 'TjaZl4T9IfrFISw3JP17IXbx2'
consumer_secret = 'pzNrHBzBT6Syx73J9SY3BipNMRWB88bm7HVJKkepRcqANnyZfO'
access_token = '1542419213075087360-HQDe3p5cJexh1Sarday2Ks1iEdndVx'
access_token_secret = '2BsC6kRITJDJVLcTCO2G2hwJmvXIue7B41Hj2XGNtE6et'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

username = 'params00'
user = api.get_user(screen_name=username)
followers_count = user.followers_count
friends_count = user.friends_count

print(f"The number of followers of {username} is: {followers_count}")
print(f"The number of following of {username} is: {friends_count}")

