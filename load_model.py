import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import tensorflow as tf

# 读取数据
data = pd.read_csv('daData_filter1.csv')
df = data[['comment', 'Sentiment', 'sentiment_fix']]
#print(df)

# 数据清洗和分词 (在本地环境中使用jieba进行分词)
def preprocess_text(text):
    # 清除特殊字符和标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    # words = jieba.cut(text)
    # return ' '.join(words)
    return text  # 暂时返回未分词的文本

df['processed'] = df.iloc[:, 0].apply(preprocess_text)

# 特征提取
tokenizer = Tokenizer(num_words=10000)  # 词汇量设置为10000
print(tokenizer)
tokenizer.fit_on_texts(df['processed'])
sequences = tokenizer.texts_to_sequences(df['processed'])
data = pad_sequences(sequences, maxlen=200)

# 加载模型
loaded_model = tf.keras.models.load_model('Iemo2')
  
# 预测评论的情感
df['sentiment_predict_trained'] = loaded_model.predict(data)[:,0]
df.to_csv('2.csv', index=False)