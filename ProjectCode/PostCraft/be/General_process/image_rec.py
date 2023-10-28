import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import openai
import os

openai.api_key = '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'
nltk.download('stopwords')


class TextSimilarityFinder:
    def __init__(self, data_path):
        self.data = pd.read_excel(data_path)
        self.data["Imagelabels"] = self.data["Imagelabels"].fillna('')
        self.data["Cleaned_Tweet_Content"] = self.data["Cleaned_Tweet_Content"].fillna('')
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        words = re.sub(r'[^\w\s]', '', text).split()
        words = [word.lower() for word in words if word.lower() not in self.stop_words]
        return ' '.join(words)

    def find_most_similar_text(self, input_text):
        valid_indices = self.data["Imagelabels"].apply(lambda x: isinstance(x, str) and x != '').index
        context_values = self.data["Cleaned_Tweet_Content"][valid_indices].tolist()
        imagelabels = self.data["Imagelabels"][valid_indices]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(context_values)
        user_input = self.preprocess_text(input_text)
        user_tfidf = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_tfidf, tfidf_matrix)
        most_similar_index = similarities.argmax()

        if most_similar_index < len(context_values):
            most_similar_imagelabel = imagelabels.iloc[most_similar_index]
            most_similar_text = context_values[most_similar_index]
            return most_similar_text, most_similar_imagelabel
        else:
            return None, None


    def generate_imagekey(self,inputs):
        prompt = "Describe a image you think is suitable for the Twitter post . The content of the image must embody the following " \
                 "characteristics:{}.You may not use all of the characteristics, you should make the image normal." \
                 " The ouput can only consists of the content. you don't need to talk to me. you ouput can not consists any " \
                 "web link or punctuation marks or other mark. your answer should within 50 words"\
            .format(inputs)

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )

        generated_imagekey = response['choices'][0]['text'].strip()
        return generated_imagekey

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "Processed_Tweets.xlsx")
# data_path = "/Users/liusiyan/PycharmProjects/IRS5001/Processed_Tweets.xlsx"
input_text = 'girl, red ,lipstick'

similarity_finder = TextSimilarityFinder(data_path)
most_similar_text, most_similar_imagelabel = similarity_finder.find_most_similar_text(input_text)

if most_similar_text and most_similar_imagelabel:
    # print("Recommended image labels:", most_similar_imagelabel)
    imagekey = similarity_finder.generate_imagekey(most_similar_imagelabel)
else:
    imagekey = similarity_finder.generate_imagekey(input_text)
# print(imagekey)
