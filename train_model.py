import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from snownlp import SnowNLP  # 确保安装了SnowNLP
import jieba

# 读取数据
data = pd.read_csv('dmData_Marry_train.csv')
df = data[['comment', 'sentiment']]

# 数据清洗和分词 (在本地环境中使用jieba进行分词)
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    if not text.strip():
        return '无内容'  # 返回一个默认的非None字符串
    return text

df['processed'] = df['comment'].apply(preprocess_text)

def sentiment_label(text):
    try:
        if text is None or text == '无内容':  # 检查是否为None或无内容
            return 0  # 或者返回一个默认的情感标签
        s = SnowNLP(text)
        if s.sentiments < 0.7:
            return 0 
        elif s.sentiments < 0.9:
            return 0.5
        else:
            return 1
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return 0  # 或者返回一个默认的情感标签

df['sentiment_fix'] = df['processed'].apply(sentiment_label)

# 特征提取
tokenizer = Tokenizer(num_words=10000)  # 词汇量设置为10000
tokenizer.fit_on_texts(df['processed'])
sequences = tokenizer.texts_to_sequences(df['processed'])
data = pad_sequences(sequences, maxlen=200)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, df['sentiment_fix'], test_size=0.2, random_state=42)

# 构建LSTM模型
model = Sequential()
model.add(Embedding(10000, 128, input_length=200))  # 嵌入维度设置为128
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  # LSTM单元数设置为128
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)  # 运行5个epoch

# 保存模型
model.save('Iemo')

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')