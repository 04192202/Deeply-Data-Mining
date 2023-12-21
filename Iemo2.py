import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pkuseg  # 导入 pkuseg

# 初始化 pkuseg 分词器
seg = pkuseg.pkuseg()

# 读取数据
data = pd.read_csv('dmData_Marry_train.csv')
df = data[['comment', 'Sentiment','Confidence']]

# 数据清洗和分词 (使用 pkuseg 进行分词)
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    if not text.strip():
        return '无内容'  # 返回一个默认的非None字符串
    # 使用 pkuseg 进行分词
    return ' '.join(seg.cut(text))

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
df.to_csv('daData_filter2.csv', index=False)

# 特征提取
tokenizer = Tokenizer(num_words=10000)  # 词汇量设置为10000
tokenizer.fit_on_texts(df['processed'])
sequences = tokenizer.texts_to_sequences(df['processed'])
data = pad_sequences(sequences, maxlen=200)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, df['sentiment_fix'], test_size=0.2, random_state=42)

# 构建LSTM模型
model = Sequential()
model.add(Embedding(20000, 128, input_length=200))  # 嵌入维度设置为128
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  # LSTM单元数设置为128
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)  # 运行5个epoch

# 保存模型
model.save('Iemo3')

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')








import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from tensorflow.keras.models import load_model

# 加载模型
model = load_model('Iemo3')  # 替换成你的模型文件路径

# 预测
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC 曲线和 AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# 准确度、精确度、召回率和 F1 分数可视化
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

metrics_values = [accuracy, precision, recall, f1]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple'])
plt.ylabel('Score')
plt.title('Performance Metrics')
plt.show()

# 学习曲线
train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training Score')
plt.plot(train_sizes, test_scores_mean, label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.show()