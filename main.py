# 添加库函数
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime,date
from sqlalchemy import create_engine, text
import pymysql
import jieba
import wordcloud
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
import snownlp
import seaborn as sn
if 0:
    # --------------------------------------------------------------------------------------------------------------------#
    # ---------------------------------------------  1.数据爬取  ----------------------------------------------------------#
    # --------------------------------------------------------------------------------------------------------------------#
    try:
        # 发送请求
        # cid：视频对应cid
        response = requests.get("https://comment.bilibili.com/cid.xml")
        # 检查请求是否成功
        if response.status_code != 200:
            print("请求失败，状态码：", response.status_code)
            exit()
        # 设置响应编码
        response.encoding = 'utf8'
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # 提取所有包含弹幕的标签
        all_txt = soup.find_all("d")
        # 提取弹幕属性和内容
        para = [danmaku.attrs["p"] for danmaku in all_txt]
        texts = [danmaku.get_text(strip=True) for danmaku in all_txt]
        text_count = len(texts)
        #print(f"弹幕数: {text_count}")
    except requests.RequestException as e:
        print("请求过程中发生错误：", e)
    # --------------------------------------------------------------------------------------------------------------------#
    # ----------------------------------------------  2.数据预处理  -------------------------------------------------------#
    # --------------------------------------------------------------------------------------------------------------------#
    # 假设 para 和 texts 分别其他属性字段与评论变量 放入dataframe中
    list_danmu = pd.DataFrame({"paragraphs": para, "comment": texts})
    # 数据处理部分 re 将弹幕的第一个属性值拆分为time_happen列
    list_danmu["time_happen"] = list_danmu['paragraphs'].map(lambda x: x.split(',')[0])
    # 将弹幕的第二个属性值拆分为danmu_location列
    list_danmu["danmu_location"] = list_danmu['paragraphs'].map(lambda x: x.split(',')[1])
    # 将弹幕的第三个属性值拆分为danmu_size列
    list_danmu["danmu_size"] = list_danmu['paragraphs'].map(lambda x: x.split(',')[2])
    # 将弹幕的第四个属性值拆分为danmu_colorg列
    list_danmu["danmu_color"] = list_danmu['paragraphs'].map(lambda x: x.split(',')[3])
    # 将弹幕的第五个属性值拆分为danmu_ture_time列
    list_danmu["danmu_ture_time"] = list_danmu['paragraphs'].map(lambda x: x.split(',')[4])
    # 将弹幕的第六个属性值拆分为danmu_mode列
    list_danmu["danmu_mode"] = list_danmu['paragraphs'].map(lambda x: x.split(',')[5])
    # 将弹幕的第七个属性值拆分为user_id列
    list_danmu["user_id"] = list_danmu['paragraphs'].map(lambda x: x.split(',')[6])
    # 将时间戳先转换为datetime
    list_danmu["danmu_day_time"] = list_danmu['danmu_ture_time'].apply(lambda x: datetime.fromtimestamp(int(x)))
    # 将datetime转换为date日期
    list_danmu["danmu_day"] = list_danmu['danmu_day_time'].apply(lambda x: x.date().strftime("%Y/%m/%d"))
    # 将datetime转换为hour小时
    list_danmu["danmu_time_hour"] = list_danmu['danmu_day_time'].apply(lambda x: x.time().strftime("%H"))
    # 提取月份和日期信息
    list_danmu['danmu_time_month'] = list_danmu['danmu_day_time'].apply(lambda x: x.strftime("%m"))
    list_danmu['danmu_time_day'] = list_danmu['danmu_day_time'].apply(lambda x: x.strftime("%d"))
    # 原始数据保存至csv
    list_danmu.to_csv('./group_project/Coding/dataset/dmData_marry_yuanshi.csv', encoding='utf-8', index=False)

    # 数据库连接
    try:
        engine = create_engine('mysql+pymysql://root:123456@localhost/dm_sql')
        # 检查表是否存在
        with engine.connect() as conn:
            result = conn.execute("SHOW TABLES LIKE 'dm_list_marry';")
            table_exists = result.fetchone() is not None
        # 根据表是否存在选择不同的操作
        if table_exists:
            # 表存在，追加数据
            list_danmu.to_sql(name='dm_list_marry', con=engine, index=False, if_exists='replace')
        else:
            # 表不存在，创建新表并插入数据
            list_danmu.to_sql(name='dm_list_marry', con=engine, index=False, if_exists='replace')
    except (pymysql.Error, Exception) as error:
        print("An error occurred:", error)
    finally:
        engine.dispose()
    # --------------------------------------------------------------------------------------------------------------------#
    # ----------------------------------------------  3.词频分析图  -------------------------------------------------------#
    # ----------------------------------------------  3.1 词云图  -------------------------------------------------------#
    # --------------------------------------------------------------------------------------------------------------------#
    # 指定中文字体路径 (需要根据自己的系统起定义)
    font_path = 'simsun.ttc'
    # 将弹幕列表转换为单一字符串
    danmustr = ''.join(texts)
    # 使用jieba进行中文分词
    words = list(jieba.cut(danmustr))
    # 过滤长度为1的词和停用词
    filtered_words = [word for word in words if len(word) > 1 and word not in STOPWORDS]
    #filtered_words = [i for i in words if len(i)>3]
    print(filtered_words)
    # 创建词云对象，设置背景颜色、最大词数等
    #ground = np.array(Image.open('./may.jpg'))
    wc = wordcloud.WordCloud(
        font_path=font_path,
        background_color='black',
        max_words=200,
        width=1000,
        height=1000)
    # 生成词云
    wc.generate(' '.join(filtered_words))
    plt.imshow(wc)
    plt.show()
    # 保存词云为图像文件
    wc.to_file('./group_project/Coding/figure/wordcloud.png')
    # --------------------------------------------------------------------------------------------------------------------#
    # ----------------------------------------- 3.2 关键词词频分析图  -----------------------------------------------------#
    # --------------------------------------------------------------------------------------------------------------------#
    # 获取关键词和频率
    keywords_frequency = wc.words_
    # 按频率降序排序关键词和频率
    sorted_keywords_frequency = sorted(keywords_frequency.items(), key=lambda x: x[1], reverse=True)
    # 仅保留前20名
    top_15_keywords_frequency = sorted_keywords_frequency[:15]

    # 提取关键词和频率
    keywords = []
    frequencys = []
    for keyword, frequency in top_15_keywords_frequency:
        keywords.append(keyword)
        frequency = round(frequency, 3)
        frequencys.append(frequency)

    # 绘图
    matplotlib.rcParams['font.family']='SimHei'
    matplotlib.rcParams['font.size']=12

    # 创建子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制雷达图
    axs[0] = plt.subplot(1, 2, 1, polar=True)
    theta = np.linspace(0, 2 * np.pi, len(keywords), endpoint=False)
    axs[0].fill(theta, frequencys, alpha=0.5, color='red')
    axs[0].set_xticks(theta)
    axs[0].set_xticklabels(keywords)
    axs[0].set_title('关键词频率雷达图')
    axs[0].set_rlabel_position(335)  # 设置刻度标签的位置

    # 绘制折线图
    axs[1].plot(keywords, frequencys, marker='o', linestyle='-', color='cornflowerblue')
    axs[1].tick_params(axis='x', rotation=30)
    axs[1].set_title('关键词频率折线图')
    axs[1].set_xlabel('关键词')
    axs[1].set_ylabel('词频')

    plt.tight_layout()
    plt.savefig('./group_project/Coding/figure/关键词词频分析图.png')
    plt.show()
df = pd.read_csv('./group_project/Coding/dataset/dmData_marry_yuanshi.csv', encoding='utf-8')
# 所有弹幕数据按照发送时间排序
df = df.sort_values('danmu_ture_time', ascending=True)
# --------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------  4.分析弹幕数量和日期的关系  ------------------------------------------------#
# ----------------------------------------  4.1 分析不同日期的弹幕数量  ------------------------------------------------#
# --------------------------------------------------------------------------------------------------------------------#
# 创建日期列表
list_time=df["danmu_day"].tolist()
days = []
days_simple = []
for i in list_time:
    if i not in days:
        days.append(i)
        days_simple.append(i[5: ])
# 弹幕每日数量统计
danmu_counts=df["danmu_day"].value_counts()
danmu_counts_dic=dict(danmu_counts)
danmu_counts_list=[]
for i in days:
    danmu_counts_list.append(danmu_counts_dic[i])
'''
# 创建DataFrame
df_result = pd.DataFrame({'days': days_simple, 'danmu_counts_list': danmu_counts_list})
# 将DataFrame保存为CSV文件
df_result.to_csv('result.csv', index=False)
'''
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['font.size']=12

# 创建一个函数判断是否为周末
def is_weekend(date_str):
    # 将日期字符串转换为日期对象，假设年份为当前年份
    date_object = datetime.strptime(f"{datetime.now().year}/{date_str}", "%Y/%m/%d")
    # 判断是否是周末（星期六或星期天）
    if date_object.weekday() in [5, 6]:
        return True
    else:
        return False
    
# 判断是否为周末，创建颜色列表
colors_list1 = []
for i in days_simple:
    if is_weekend(i):
        colors_list1.append("coral")
    else:
        colors_list1.append("cornflowerblue")

# 绘制不同日期的弹幕数量统计柱状图
plt.figure(figsize=(16, 8))
plt.bar(days_simple, danmu_counts_list,color=colors_list1,width=0.4, alpha = 0.6)
plt.plot(danmu_counts_list,color='r',linewidth=1)
plt.title("不同日期的弹幕数量统计图(视频发布于2023年10月23日)")
plt.xlabel("发送弹幕日期")
plt.ylabel("弹幕数量")
plt.xticks(rotation=45)
for i, v in enumerate(danmu_counts_list):
    if v >= 150:
        plt.text(i, v+10, str(round(v,2)), ha='center')
# 创建自定义图例对象  
legend_handles = []  
for color in ["coral", "cornflowerblue"]:
    if color ==  "coral":
        line = plt.Line2D([], [], color=color, label='weekend')  # 创建一个只有颜色的Line2D对象  
    else:
        line = plt.Line2D([], [], color=color, label='weekday')  # 创建一个只有颜色的Line2D对象  
    legend_handles.append(line)
# 创建图例并显示在图表上
plt.legend(handles=legend_handles, loc='upper right')
plt.tight_layout()  # 调整子图之间的间距
plt.savefig("./group_project/Coding/figure/不同日期的弹幕数量统计图.png", bbox_inches='tight')
plt.show()
# --------------------------------------------------------------------------------------------------------------------#
# --------------------------------------  4.2 分析上午、中午和下午的弹幕数量  -------------------------------------------#
# --------------------------------------------------------------------------------------------------------------------#
# 统计早午晚弹幕数量
danmu_hour_counts=df["danmu_time_hour"].value_counts()
danmu_hour_counts_dic=dict(danmu_hour_counts)
danmu_time_list = [0, 0, 0, 0]
for i in danmu_hour_counts_dic.keys():
    ii = int(i)
    if ii > 6 and ii <= 12:
        danmu_time_list[1] += danmu_hour_counts_dic[i]
    elif ii > 12 and ii <= 18:
        danmu_time_list[2] += danmu_hour_counts_dic[i]
    elif ii > 18 and ii <= 24:
        danmu_time_list[3] += danmu_hour_counts_dic[i]
    else:
        danmu_time_list[0] += danmu_hour_counts_dic[i]

# 绘制早午晚弹幕数量统计图
plt.figure(figsize=(16, 8))

# 子图1：柱状图
plt.subplot(1, 2, 1)
label = ['night(0-6)', 'morning(6-12)', 'afternoon(12-18)', 'evening(18-24)']
plt.barh(label, danmu_time_list, color=["cornflowerblue", "cornflowerblue", "coral", "coral"], height=0.4, alpha = 0.6)
plt.plot([np.mean(danmu_time_list)] * len(label), label, color='r', linewidth=2)
plt.text(np.mean(danmu_time_list) + 20, label[0], "mean:" + str(int(np.mean(danmu_time_list))), ha='left')
plt.title("早午晚弹幕数量统计图")
plt.xlabel("弹幕数量")
plt.ylabel("时间段")
#plt.xticks(rotation=30)
for i, v in enumerate(danmu_time_list):
    plt.text(v + 10, i, str(round(v, 2)), ha='center')

# 子图2：饼状图
plt.subplot(1, 2, 2)
# 绘制饼状图
matplotlib.rcParams['font.size']=10
colors_list = ["cornflowerblue", "orange", "turquoise", "crimson", \
               "indigo", "hotpink", "navy", "coral", "limegreen", "silver", "gold", "chocolate"]
patches, texts, autotexts = plt.pie(danmu_time_list, labels=label, autopct='%3.1f%%', colors = colors_list,
        pctdistance=1.1, labeldistance = None, startangle = 30, explode=[0, 0, 0.1, 0.05], 
        radius = 0.9, counterclock = True)
# 添加统一图例
plt.legend(patches, label, loc="lower center", bbox_to_anchor=(0.5, -0.2))
plt.title("早午晚弹幕数量饼状图")
# 保存整体图像为图片文件
plt.savefig("./group_project/Coding/figure/早午晚弹幕数量统计图.png")
plt.show()
# --------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------  4.3 分析不同时间段的弹幕数量  ----------------------------------------------#
# --------------------------------------------------------------------------------------------------------------------#
# 统计各时间段弹幕数量
danmu_hour_counts=df["danmu_time_hour"].value_counts()
danmu_hour_counts_dic=dict(danmu_hour_counts)
danmu_hour_counts_list=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in danmu_hour_counts_dic.keys():
    ii = int(i)
    if ii > 0 and ii <= 2:
        danmu_hour_counts_list[0] += danmu_hour_counts_dic[i]
    elif ii > 2 and ii <= 4:
        danmu_hour_counts_list[1] += danmu_hour_counts_dic[i]
    elif ii > 4 and ii <= 6:
        danmu_hour_counts_list[2] += danmu_hour_counts_dic[i]
    elif ii > 6 and ii <= 8:
        danmu_hour_counts_list[3] += danmu_hour_counts_dic[i]
    elif ii > 8 and ii <= 10:
        danmu_hour_counts_list[4] += danmu_hour_counts_dic[i]
    elif ii > 10 and ii <= 12:
        danmu_hour_counts_list[5] += danmu_hour_counts_dic[i]
    elif ii > 12 and ii <= 14:
        danmu_hour_counts_list[6] += danmu_hour_counts_dic[i]
    elif ii > 14 and ii <= 16:
        danmu_hour_counts_list[7] += danmu_hour_counts_dic[i]
    elif ii > 16 and ii <= 18:
        danmu_hour_counts_list[8] += danmu_hour_counts_dic[i]
    elif ii > 18 and ii <= 20:
        danmu_hour_counts_list[9] += danmu_hour_counts_dic[i]
    elif ii > 20 and ii <= 22:
        danmu_hour_counts_list[10] += danmu_hour_counts_dic[i]
    else:
        danmu_hour_counts_list[11] += danmu_hour_counts_dic[i]

# 绘制不同时段的弹幕数量统计图
plt.figure(figsize=(16, 8))
color=["cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue", \
        "coral", "cornflowerblue", "cornflowerblue", "cornflowerblue", "coral", "cornflowerblue", "coral"]
label = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12', '12-14', '14-16', '16-18', '18-20', '20-22', '22-24']
for i in range(len(danmu_hour_counts_list)):
    area = danmu_hour_counts_list[i]*10
    plt.scatter(label[i], danmu_hour_counts_list[i], s=area,c=color[i], alpha = 0.6)
    plt.text(label[i], danmu_hour_counts_list[i], str(danmu_hour_counts_list[i]), ha='center', va='center', fontsize=10) 

# 显示图形
plt.title('不同时间段的弹幕数量统计图')
plt.xlabel('时间段')
plt.ylabel('弹幕数量')
plt.xticks(rotation = 35)
# 保存整体图像为图片文件
plt.savefig("./group_project/Coding/figure/不同时间段的弹幕数量统计图.png")
plt.show()
# --------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------  5.1 视频中各个片段的弹幕数量  -----------------------------------------------#
# --------------------------------------------------------------------------------------------------------------------#
# 分段长度,单位为秒
fragmentLength=4
# 视频长度,单位为秒
videoTime=540
# 视频分段数
fragmentCount=int(videoTime/fragmentLength)
# 统计各个分片内的弹幕数量
timeHappenList=df["time_happen"].tolist()
countTimeHappenList=[]
for i in range(fragmentCount):
    count_time_happen = 0
    for j in range(len(timeHappenList)):
        if i * fragmentLength <= float(timeHappenList[j]) < (i + 1) * fragmentLength:
            timeHappenList[j]=i
            count_time_happen +=1
    countTimeHappenList.append(count_time_happen)

# 绘图
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['font.size']=12
plt.figure(figsize=(16, 8))
for i in range(len(countTimeHappenList)):
    area = countTimeHappenList[i]*30
    plt.scatter(i, countTimeHappenList[i], s=area,c='cornflowerblue', alpha = 0.6)

plt.xlim(0, )
plt.ylim(0, )
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(25))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(10))

plt.title('分析视频各个片段出现的弹幕数量')
plt.xlabel('视频的各个片段(区间长度为4秒)')
plt.ylabel('视频片段中的弹幕数量')
plt.xticks(rotation = 35)
# 保存整体图像为图片文件
plt.savefig("./group_project/Coding/figure/视频各片段弹幕数量图.png")
plt.show()
# --------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------------  5.2 lstm弹幕数量预测  -----------------------------------------------#
# --------------------------------------------------------------------------------------------------------------------#
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# 读取弹幕数量数据
df_train=pd.read_csv("./group_project/Coding/dataset/danmu_number.csv")
# 拆分成训练集和测试集
# 训练集大小
trainNum=50
trainingSet = df_train.iloc[:trainNum, 1:2].values
testSet = df_train.iloc[trainNum:, 1:2].values

# 构建滞后1期阶的输入特征
# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))

# 创建具有3个时间步长和1个输出的数据结构
# 时间步长
timeSteps=3
trainingSetScaled = sc.fit_transform(trainingSet)
X_train = []
y_train = []
for i in range(timeSteps, trainNum):
    X_train.append(trainingSetScaled[i-timeSteps:i, 0])
    y_train.append(trainingSetScaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 用50个神经元和4隐藏层构建LSTM。
# 在输出层中分配1个神经元以预测标准化弹幕数。
# 使用 MSE 损失函数和 Adam 随机梯度下降优化器。

# 添加第一层 LSTM 和一些 Dropout 正则化
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

# 添加第二层 LSTM 和一些 Dropout 正则化
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))

# 添加第三层 LSTM 和一些 Dropout 正则化
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))

# 添加第四层 LSTM 和一些 Dropout 正则化
model.add(Dropout(0.2))
model.add(LSTM(units = 50))

# 添加输出层
model.add(Dropout(0.2))
model.add(Dense(units = 1))

# Adam 是一种常见的优化器，它被认为在大多数情况下都是一个合适的选择。它使用自适应学习率的方法来调整模型的权重，从而使训练过程收敛到最优解
# 使用了mean_squared_error作为lossFunction
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 拟合训练
model.fit(X_train, y_train, epochs = 4000, batch_size = 4)

# 准备测试数据
datasetTrain = df_train.iloc[:50, 1:2]
datasetTest = df_train.iloc[50:, 1:2]
datasetTotal = pd.concat((datasetTrain, datasetTest), axis = 0)
inputs = datasetTotal[len(datasetTotal) - len(datasetTest) - timeSteps:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
y_test = datasetTest.values
for i in range(timeSteps, len(datasetTotal) - len(datasetTrain) + timeSteps):
    X_test.append(inputs[i-timeSteps:i, 0])
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



# 使用测试集进行预测
predicted_danmuCount = model.predict(X_test)
predicted_danmuCount = sc.inverse_transform(predicted_danmuCount)



# 评价指标
# 评估结果的均方误差（Mean Squared Error）和平均绝对误差（Mean Absolute Error）
mse = mean_squared_error(y_test, predicted_danmuCount)
mae = mean_absolute_error(y_test, predicted_danmuCount)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")



# 将结果进行可视化
plt.plot(df_train.loc[50:, 'datetime'],datasetTest.values, color = 'red', label = '实际弹幕数量')
plt.plot(df_train.loc[50:, 'datetime'],predicted_danmuCount, color = 'blue', label = '预测弹幕数量')
plt.xticks(np.arange(0,5,1))
plt.title('弹幕数量预测')
plt.xlabel('日期')
plt.ylabel('弹幕数量')
# 使用微软雅黑字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 处理负号显示异常
plt.rcParams['axes.unicode_minus'] = False
plt.legend()
plt.savefig("./group_project/Coding/figure/弹幕数量预测.png")
plt.show()

# --------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------  6. 视频整体弹幕情感分析  -------------------------------------------------#
# --------------------------------------------  6.1 分析整体弹幕情绪  --------------------------------------------------#
# --------------------------------------------------------------------------------------------------------------------#
# 计算snowNLP
df_snownlp = pd.read_csv('./group_project/Coding/dataset/dmData_Marry_baiduNLP.csv')
df_snownlp = df_snownlp[['comment','sentiment_baidunlp']]
df_snownlp['sentiment_snownlp']=df_snownlp["comment"].apply(lambda x:snownlp.SnowNLP(x).sentiments)
# Iemo
df_Iemo = pd.read_csv('./group_project/Coding/dataset/dmData_Marry_Iemo.csv')
df_snownlp['sentiment_Iemo'] = df_Iemo['sentiment_fix']
# 将情感得分存至csv
print(df_snownlp.shape)
df_snownlp.to_csv('./group_project/Coding/dataset/dmData_Marry_final.csv', encoding='utf-8', index=False)
##########
#df_snownlp = pd.read_csv('./group_project/dataset/dmData_May_finalNLP.csv')
##########
# 设置字体为中文
plt.rcParams['font.family'] = 'SimHei'
# 设置字体大小
plt.rcParams['font.size'] = 12
# 创建一个包含两个子图的大图
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
# 绘制第一个子图（snownlp）
ax1 = sn.distplot(df_snownlp['sentiment_snownlp'], hist_kws={'color': 'deepskyblue'}, kde_kws={'color': 'orange'}, bins=20, ax=axes[0])
ax1.set_title("Snownlp情感倾向")
# 绘制第二个子图（baidunlp）
ax2 = sn.distplot(df_snownlp['sentiment_baidunlp'], hist_kws={'color': 'deepskyblue'}, kde_kws={'color': 'orange'}, bins=20, ax=axes[1])
ax2.set_title("Baidunlp情感倾向")
# 绘制第三个子图（Iemo）
ax3 = sn.distplot(df_snownlp['sentiment_Iemo'], hist_kws={'color': 'deepskyblue'}, kde_kws={'color': 'orange'}, bins=20, ax=axes[2])
ax3.set_title("Iemo情感倾向")
# 调整子图之间的距离
plt.tight_layout()
# 保存整体图像为图片文件
plt.savefig("./group_project/Coding/figure/整体弹幕情感.png")
# 显示整体图像
plt.show()
# --------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------  6.2 分析各个情绪种类的弹幕数量  ---------------------------------------------#
# --------------------------------------------------------------------------------------------------------------------#
# snowNLP
sentiment_list1=df_snownlp['sentiment_snownlp'].tolist()
count_snownlp=[0,0,0]
for i in range(len(sentiment_list1)):
    if sentiment_list1[i]>=0.6:
        count_snownlp[0]=count_snownlp[0]+1
    elif sentiment_list1[i]<0.4:
        count_snownlp[2]=count_snownlp[2]+1
    else:
        count_snownlp[1]=count_snownlp[1]+1
# baiduNLP
sentiment_list2=df_snownlp['sentiment_baidunlp'].tolist()
count_baidunlp=[0,0,0]
for i in range(len(sentiment_list2)):
    if sentiment_list2[i]==2:
        count_baidunlp[0]=count_baidunlp[0]+1
    elif sentiment_list2[i]==0:
        count_baidunlp[2]=count_baidunlp[2]+1
    else:
        count_baidunlp[1]=count_baidunlp[1]+1
# Iemo
sentiment_list3=df_snownlp['sentiment_Iemo'].tolist()
count_iemo=[0,0,0]
for i in range(len(sentiment_list3)):
    if sentiment_list3[i]==1:
        count_iemo[0]=count_iemo[0]+1
    elif sentiment_list3[i]==0:
        count_iemo[2]=count_iemo[2]+1
    else:
        count_iemo[1]=count_iemo[1]+1

# 设置字体为中文
plt.rcParams['font.family'] = 'SimHei'
# 设置字体大小
plt.rcParams['font.size'] = 12
# 假设count_snownlp和count_baidunlp是两个包含情感数量的列表
bar_width = 0.3
index = range(3)
# 使用一个柱状图显示两个情感分析工具的数据
plt.bar(index, count_snownlp, width=bar_width, label='Snownlp', alpha=0.7)
plt.bar([i + bar_width for i in index], count_baidunlp, width=bar_width, label='Baidu NLP', alpha=0.7)
plt.bar([i + 2 * bar_width for i in index], count_iemo, width=bar_width, label='Iemo', alpha=0.7)
# 设置图表标题和坐标轴标签
plt.title("整体观众的情感倾向")
plt.xlabel("情感种类")
plt.ylabel("各情感种类的弹幕数量")
# 设置x轴刻度和标签
plt.xticks([i + bar_width / 2 for i in index], ['积极', '中立', '消极'])
# 显示图例
plt.legend()
# 保存图表为图片文件
plt.savefig("./group_project/Coding/figure/情感分析对比图.png")
# 显示图表
plt.show()