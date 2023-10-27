import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_excel('Processed_Tweets.xlsx')

# 填充数值型列中的缺失值为0
numeric_columns = ['Tweet_Number_of_Reviews', 'Tweet_Number_of_Retweets', 'Tweet_Number_of_Likes',
                   'Tweet_Number_of_Looks', 'content_str_len', 'sentiment_number', 'Mention_Count',
                   'Num_Hashtags', 'Word_Count', 'Paragraph_Count']
for column in numeric_columns:
    df[column].fillna(0, inplace=True)


import seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取处理后的数据
# 选择所有数值型列进行相关性分析
numeric_columns = df.select_dtypes(include=['number'])

# 计算相关性矩阵
correlation_matrix = numeric_columns.corr()

# 绘制热力图，并旋转坐标轴标签
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Correlation Heatmap of Numeric Columns')
plt.tight_layout()
# plt.show()





corpus = df['Cleaned_Tweet_Content'].tolist()


# # 分词和去除停用词
# stop_words = set(stopwords.words('english'))
# tokenized_corpus = []
#
# for text in corpus:
#     words = word_tokenize(text)
#     words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
#     tokenized_corpus.append(words)

from collections import Counter

# 统计词频
# word_freq = Counter(word for sublist in tokenized_corpus for word in sublist)
# threshold = 2
# low_freq_words = [word for word, freq in word_freq.items() if freq <= threshold]
#
# filtered_corpus = []
# removed_words = []
# for text in tokenized_corpus:
#     filtered_text = [word for word in text if word not in low_freq_words]
#     removed_words.extend([word for word in text if word not in filtered_text])
#     filtered_corpus.append(filtered_text)
#
# print("Removed words:", removed_words)
# print(filtered_corpus[:2])

# # 保存处理后的结果到新的Excel文件
# df.to_excel('Processed_Tweets_Cleaned.xlsx', index=False)

from gensim import corpora
dictionary = corpora.Dictionary(corpus)
print(dictionary)
corpus = [dictionary.doc2bow(text) for text in corpus]
print(corpus)
