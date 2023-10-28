import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def Get_data():
    df = pd.read_excel('/Users/liusiyan/PycharmProjects/IRS5001/Processed_Tweets.xlsx')
    return df
def Plot_interaction(df):
    interaction_index = df[['Tweet_Number_of_Reviews', 'Tweet_Number_of_Retweets', 'Tweet_Number_of_Likes']].sum(axis=1)
    df['Interaction_index'] = interaction_index
    plt.figure(figsize=(10, 6))
    plt.hist(np.log1p(df['Interaction_index']), bins=100, color='skyblue', edgecolor='black')
    plt.xlabel('Log(Interaction Index)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Log(Interaction Index)')
    plt.grid(True)
    plt.show()


def Clean_Imagelabels(df):
    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    df['Imagelabels'] = df['Imagelabels'].fillna('').apply(remove_punctuation)


def Combine_textcontetn(df):
    df['Cleaned_Tweet_Content'] = df['Cleaned_Tweet_Content'] + ' ' + df['Imagelabels']


def Drop_topKOL(df):
    df = df[df['Cluster_Label'] != 2]


def Get_cold_data(df):
    columns_to_keep = [ 'Tweet_Timestamp', 'Tweet_Number_of_Reviews', 'Tweet_Number_of_Retweets',
                       'Tweet_Number_of_Likes',
                       'Tweet_Number_of_Looks', 'Followers_Count', 'Friends_Count', 'Hashtags_Content', 'Cluster_Label']
    cold_data = df.drop(columns=columns_to_keep)
    cold_data = cold_data[cold_data['Interaction_index'] > 100]
    cold_data.to_excel('cold_data.xlsx', index=False)


def Get_cold_data_full(df):
    columns_to_keep_full = ['Keyword', 'Tweet_Timestamp', 'Tweet_Number_of_Reviews', 'Tweet_Number_of_Retweets',
                            'Tweet_Number_of_Likes',
                            'Tweet_Number_of_Looks', 'Hashtags_Content', 'Cluster_Label']
    cold_data = df.drop(columns=columns_to_keep_full)
    cold_data = cold_data[cold_data['Interaction_index'] > 100]
    cold_data.to_excel('cold_data_full.xlsx', index=False)


df=Get_data()
Plot_interaction(df=df)
Clean_Imagelabels(df=df)
Combine_textcontetn(df=df)
Drop_topKOL(df=df)
Get_cold_data(df=df)
Get_cold_data_full(df=df)
