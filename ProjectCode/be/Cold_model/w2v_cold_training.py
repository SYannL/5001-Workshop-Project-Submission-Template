from gensim.models import Word2Vec
import pandas as pd
import jieba

df = pd.read_excel('cold_data.xlsx')
df['Cleaned_Tweet_Content'] = df['Cleaned_Tweet_Content'].apply(lambda x: ' '.join(jieba.cut(str(x))))

sentences = df['Cleaned_Tweet_Content'].apply(lambda x: x.split()).tolist()

word2vec_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, sg=0)

word2vec_model.save('word2vec_model_cold.bin')