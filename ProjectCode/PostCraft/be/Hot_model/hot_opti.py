import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
openai.api_key = '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'

class TweetGenerator:
    def __init__(self, data_file_path):
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.df = pd.read_excel(data_file_path)
        self.df['Cleaned_Tweet_Content'] = self.df['Cleaned_Tweet_Content'].fillna('')
        self.word2vec_model = Word2Vec.load('word2vec_model_{}_{}.bin'.format(start_date, end_date))

    def preprocess_text(self, text):
        words = str(text).split()
        return words

    def clean_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
            text = re.sub(r'\W', ' ', text)
            text = re.sub(r'\d', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            text = ' '.join(word for word in text.split() if word not in self.stop_words)
            return text
        else:
            return ""

    def find_similar_articles(self, input_text, top_n=1):
        cleaned_input_text = self.clean_text(input_text)
        input_words = self.preprocess_text(cleaned_input_text)
        all_texts = self.df['Cleaned_Tweet_Content'].apply(self.preprocess_text)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([' '.join(words) for words in all_texts])
        input_tfidf = vectorizer.transform([' '.join(input_words)])
        similarities = cosine_similarity(input_tfidf, tfidf_matrix)[0]
        similar_indices = similarities.argsort()[-top_n:][::-1]
        similar_articles = self.df.iloc[similar_indices]
        return similar_articles

    def generate_tweets(self, input_text, input_category):
        similar_articles = self.find_similar_articles(input_text)
        feature_list = similar_articles[
            ['Keyword','content_str_len', 'sentiment_number', 'Mention_Count', 'Num_Hashtags', 'Top_3_Keywords', 'Word_Count',
            'Image_Count', 'Paragraph_Count']].values.tolist()

        generated_tweets = []
        features = feature_list[0]
        feature_dict = {
            'item category': input_category,
            'the string length of content': features[1],
            'the sentiment of content (-1for negative and 1 for positive)': features[2],
            'the Mention number of content': features[3],
            'the number of Hashtags': features[4],
            'the top 3 Keywords of post': features[5],
            'the number of words': features[6],
            'the number of image': features[7],
            'the number of paragraph': features[8]
        }
        if feature_dict['the number of words'] <= 20:
            feature_dict['the number of words'] += 20
        tweet = self.generate_tweet(feature_dict)
        generated_tweets.append(tweet)
        generated_tweets.append(input_category)
        generated_tweets.extend(features[1:9])
        return generated_tweets

    def generate_tweet(self, inputs):
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
input_text = "Time for a #flawlessfoundation 💁‍♀️ Check out our amazing products to help you create the perfect base. Need some tips? Our video tutorial will show you how to apply foundation for a flawless finish on all skin types! #foundation #makeup #skin"
input_category = "lipstick"

data_file_path = 'hot_data_{}_{}.xlsx'.format(start_date, end_date)
tweet_generator = TweetGenerator(data_file_path)
generated_tweets = tweet_generator.generate_tweets(input_text, input_category)

# posttext + features in a list
print(generated_tweets)
