import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class TweetProcessor:
    def __init__(self, filepath):
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.df = pd.read_excel(filepath)
        self.vectorizer = TfidfVectorizer()

    def drop_text(self):
        columns_to_drop = ['Category', 'Tweet_Website', 'Author_Name', 'Author_Web_Page_URL',
                           'Tweet_Video_URL', 'Tweet_AD', 'Tweet_Content', 'Twitter_Username', 'Tweet_Image_URL']
        self.df.drop(columns=columns_to_drop, inplace=True)

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
            return ""  #

    def extract_top_n_keywords(self, n=3):
        tfidf_matrix = self.vectorizer.fit_transform(self.df['Cleaned_Tweet_Content'])
        self.df['Top_3_Keywords'] = self.df['Cleaned_Tweet_Content'].apply(
            lambda text: self.get_top_n_keywords(text, tfidf_matrix, n)
        )

    def get_top_n_keywords(self, text, tfidf_matrix, n):
        idx = self.df[self.df['Cleaned_Tweet_Content'] == text].index[0]
        feature_array = self.vectorizer.get_feature_names()
        tfidf_sorting = tfidf_matrix[idx].toarray()[0].argsort()[::-1][:n]
        top_n_words = [feature_array[i] for i in tfidf_sorting]
        return ', '.join(top_n_words)

    def extract_hashtags(self, text):
        if pd.notna(text):
            hashtags = re.findall(r"#(\w+)", text)
            num_hashtags = len(hashtags)
            hashtags_content = ', '.join(hashtags)
            return pd.Series([num_hashtags, hashtags_content])
        else:
            return pd.Series([])

    def word_count(self, text):
        if pd.isna(text):
            return 0
        words = text.split()
        return len(words)

    def count_images(self, text):
        if pd.isna(text):
            return 0
        url_pattern = r'https://\S+'
        urls = re.findall(url_pattern, text)
        if ';' in text:
            urls = text.split(';')
            urls = [url.strip() for url in urls]
        return len(urls)

    def paragraph_count(self, text):
        if pd.isna(text):
            return 0
        paragraphs = text.split('\n')
        paragraphs = [para for para in paragraphs if para.strip()]
        return len(paragraphs)

    def count_mentions(self, text):
        if pd.notna(text):
            mentions = re.findall(r"@\w+", text)
            return len(mentions)
        else:
            return 0

    def convert_to_number(self, s):
        if pd.isna(s) or s == '':
            return 0.0
        s = str(s)
        s = s.replace(',', '')
        if 'K' in s:
            s = s.replace('K', '')
            return float(s) * 1000
        elif 'M' in s:
            s = s.replace('M', '')
            return float(s) * 1000000
        else:
            return float(s)

    def plot_elbow_method(self, max_k=10):
        self.df['Followers_Count'].fillna(0, inplace=True)
        self.df['Tweet_Number_of_Looks'].fillna(0, inplace=True)

        features = self.df[['Followers_Count', 'Tweet_Number_of_Looks']].values
        distortions = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, max_k + 1), distortions, marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method')

        best_k = 4
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        kmeans.fit(features)

        self.df['Cluster_Label'] = kmeans.labels_

        plt.subplot(1, 2, 2)
        for cluster_label in range(best_k):
            cluster_data = self.df[self.df['Cluster_Label'] == cluster_label]
            plt.scatter(cluster_data['Followers_Count'], cluster_data['Tweet_Number_of_Looks'],
                        label=f'Cluster {cluster_label}', alpha=0.5)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, color='red')
        plt.xlabel('Followers Count')
        plt.ylabel('Tweet Number of Looks')
        plt.title('K-means Clustering')
        plt.legend()
        plt.tight_layout()
        plt.show()
    def process_tweets(self):
        self.df['Mention_Count'] = self.df['Tweet_Content'].apply(self.count_mentions)
        self.df[['Num_Hashtags', 'Hashtags_Content']] = self.df['Tweet_Content'].apply(self.extract_hashtags)
        self.df['Cleaned_Tweet_Content'] = self.df['Tweet_Content'].apply(self.clean_text)
        self.extract_top_n_keywords()
        self.df['Word_Count'] = self.df['Tweet_Content'].apply(self.word_count)
        self.df['Image_Count'] = self.df['Tweet_Image_URL'].apply(self.count_images)
        self.df['Paragraph_Count'] = self.df['Tweet_Content'].apply(self.paragraph_count)
        self.df['Tweet_Number_of_Retweets'] = self.df['Tweet_Number_of_Retweets'].apply(self.convert_to_number)
        self.df['Tweet_Number_of_Likes'] = self.df['Tweet_Number_of_Likes'].apply(self.convert_to_number)
        self.df['Tweet_Number_of_Looks'] = self.df['Tweet_Number_of_Looks'].apply(self.convert_to_number)
        self.df['Tweet_Number_of_Reviews'] = self.df['Tweet_Number_of_Reviews'].apply(self.convert_to_number)
        self.drop_text()
        self.plot_elbow_method(max_k=10)
        self.df = self.df[self.df['Cluster_Label'] != 1]
        self.df = self.df.dropna(subset=['Tweet_Timestamp'])

    def save_to_excel(self, output_filepath):
        self.df.to_excel(output_filepath, index=False)


processor = TweetProcessor('Tweets_makeup_raw.xlsx')
processor.process_tweets()
processor.save_to_excel('Processed_Tweets.xlsx')
