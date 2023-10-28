from gensim.models import Word2Vec
import pandas as pd
import jieba
start_date = pd.to_datetime('2023-08-22')
end_date = pd.to_datetime('2023-09-22')
df = pd.read_excel('hot_data_{}_{}.xlsx'.format(start_date, end_date))
df['Cleaned_Tweet_Content'] = df['Cleaned_Tweet_Content'].apply(lambda x: ' '.join(jieba.cut(str(x))))

sentences = df['Cleaned_Tweet_Content'].apply(lambda x: x.split()).tolist()

# Train Word2Vec model with CBOW
word2vec_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, sg=0)

# save
word2vec_model.save('word2vec_model_{}_{}.bin'.format(start_date, end_date))