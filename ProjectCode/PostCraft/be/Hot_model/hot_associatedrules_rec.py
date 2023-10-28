from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import numpy as np
import openai
import os

openai.api_key = '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'

def associated_rules(inputtext: list[str],startdate,enddate):
    inputtext = [item for item in inputtext if item.strip() != '']

    script_dir = os.path.dirname(os.path.abspath(__file__))


    file_path = os.path.join(script_dir, "hot_data_2023-08-22 00_00_00_2023-09-22 00_00_00.xlsx")
    # print(f"Current working directory: {os.getcwd()}")
    # print(f"Does the file exist? {os.path.exists(file_path)}")


    data = pd.read_excel(file_path)
    keword=data['Keyword']
    data = data.drop(['Keyword','Imagelabels', 'Cleaned_Tweet_Content', 'Top_3_Keywords'], axis=1)

    def map_sentiment_level(sentiment_number):
        if sentiment_number > 0.2:
            return 'positive'
        elif sentiment_number < -0.2:
            return 'negative'
        else:
            return 'neutral'
    data['sentiment_number'] = data['sentiment_number'].apply(map_sentiment_level)
    # content_str_len
    num_bins = 5
    data['content_str_len'] = pd.qcut(data['content_str_len'], q=num_bins, labels=['very short', 'short', 'normal', 'long', 'very long'])
    #Mention_Count
    data['Mention_Count'] = data['Mention_Count'].apply(lambda x: 'not mentioned' if x == 0 else 'mentioned')
    # Num_Hashtag
    conditions = [
        (data['Num_Hashtags'] == 0),
        (data['Num_Hashtags'] > 0) & (data['Num_Hashtags'] < 3),
        (data['Num_Hashtags'] >= 3)
    ]
    labels = ['no hashtag', 'few hashtags', 'some hashtags']
    data['Num_Hashtags'] = np.select(conditions, labels, default='Other')
    #Word_Count
    num_bins = 3
    data['Word_Count'] = pd.qcut(data['Word_Count'], q=num_bins, labels=['less', 'medium', 'more'])
    #Image_Count
    conditions = [
        (data['Image_Count'] == 0),
        (data['Image_Count'] == 1),
        (data['Image_Count'] > 1)
    ]
    labels = ['no image', 'one image', 'some images']
    data['Image_Count'] = np.select(conditions, labels, default='Other')
    #Interaction_index
    num_bins = 3
    data['Interaction_index'] = pd.qcut(data['Interaction_index'], q=num_bins, labels=['less', 'medium', 'more'])

    onehot_data = pd.get_dummies(data, columns=['content_str_len', 'sentiment_number', 'Mention_Count',
                                      'Num_Hashtags', 'Word_Count', 'Image_Count',
                                      'Paragraph_Count', 'Interaction_index'])
    # print(onehot_data)
    user_input_df = pd.DataFrame({feature: [1] for feature in inputtext})
    frequent_itemsets = apriori(onehot_data, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
    filtered_rules3 = rules[rules["antecedents"].apply(lambda x: any(item in inputtext for item in x))]

    if filtered_rules3.empty:
        print("No matching rules found for the given input features.")
        return None

    longest_rule_index = filtered_rules3['consequents'].apply(lambda x: len(x)).idxmax()
    longest_rule = filtered_rules3.at[longest_rule_index, 'consequents']

    extended_features = inputtext + list(longest_rule)

    feature_name = []
    feature_value = []

    for item in extended_features:
        parts = item.rsplit('_', 1)
        feature_name.append(parts[0])
        feature_value.append(parts[1])

    filtered_data = data.filter(items=feature_name)

    vectorizer = CountVectorizer()
    text_vectors = vectorizer.fit_transform(filtered_data.astype(str).values.sum(axis=1))


    feature_value_text = ' '.join(feature_value)
    feature_value_vector = vectorizer.transform([feature_value_text])

    similarities = cosine_similarity(text_vectors, feature_value_vector)
    most_similar_index = similarities.argmax()

    similar_row_values = data.iloc[most_similar_index].values.tolist()
    # 将 int64 类型转换为 Python 内置整数类型
    similar_row_values = [int(value) if isinstance(value, np.int64) else value for value in similar_row_values]
    return similar_row_values


def generate_tweets(input_category):
    features=output
    # print(features)

    generated_tweets = []
    feature_dict = {
        'item category': '{}'.format(input_category),
        'the string length of content': features[0],
        'the sentiment of content (-1for negative and 1 for positive)': features[1],
        'the Mention number of content': features[2],
        'the number of Hashtags': features[3],
        'the number of words': features[4],
        'the number of image': features[5],
        'the number of paragraph': features[6]
    }
    tweet = generate_tweet(feature_dict)
    generated_tweets.append(tweet)
    return generated_tweets


def generate_tweet(inputs):
    prompt = f"Generate a high-quality tweet for the Twitter platform. :\n"
    for feature in inputs:
        prompt += f"- {feature}: {inputs[feature]}\n"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=280,
        temperature=0.7
    )

    generated_tweet = response['choices'][0]['text'].strip()
    return generated_tweet

start_date = pd.to_datetime('2023-08-22')
end_date = pd.to_datetime('2023-09-22')

user_input_features3 = ["content_str_len_very long", "sentiment_number_positive", "Num_Hashtags_few hashtags"]
output = associated_rules(user_input_features3,startdate=start_date,enddate=end_date)

input_category='lipstick'
tweet_rec=generate_tweets(input_category=input_category)[0]
# print(tweet_rec)