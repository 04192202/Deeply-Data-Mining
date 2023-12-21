

import json
from pyecharts import options as opts
from pyecharts.charts import Graph

import json
import pandas as pd
# 读取JSON文件
with open('response_2.json', 'r', encoding='utf-8') as file:
    json_data = file.read()

# 解析JSON数据
data = json.loads(json_data)
data_dict = json.loads(data['data'])


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# 提供的数据
provided_data=data_dict
relation_word_list = provided_data.get('relation_word_list', [])

data=relation_word_list


for item in data:
    item["search_index"] = int(item["search_index"])
    item["relation_score"]= int(item["relation_score"])
# 定义中心节点
    

new_data = []
for item in data:
    if 0.5 <= 1/item["relation_score"] <= 10:
        new_data.append(item)



center = {
    "name": "婚姻",
    "symbolSize": 50,
    "itemStyle": {"color": "blue"},
    "label": {"show": True, "position": "inside"}
}

nodes = [center]
for item in data:
    node = {
        "name": item["relation_word"],
        # 节点大小和搜索指数成正比
        "symbolSize": item["search_index"] / 1000000,
        # 节点颜色和相关性变化有关
        "itemStyle": {"color": "green" if item["correlation_change"] else "red"},
        "label": {"show": True},
        
        
    }
    nodes.append(node)

# 定义边
edges = []
for item in data:
    edge = {
        "source": "婚姻",
        "target": item["relation_word"],
        # 边的长度和组合指数成反比
        "value":  1 / item["relation_score"],
        "lineStyle": {"color": "grey"}
        
    }
    edges.append(edge)


# 创建一个图表实例
graph = Graph()

# 添加数据源和配置参数
graph.add(
    "",
    nodes=nodes,
    links=edges,
    repulsion=4000,
    layout= "circular",
    is_rotate_label=True
    
)


# 设置图表的标题，提示框，图例等
graph.set_global_opts(
    title_opts=opts.TitleOpts(title="关联词图示例"),
    tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b} : {c}"),
    legend_opts=opts.LegendOpts(orient="vertical", pos_left="2%", pos_top="20%")
)

# 生成一个HTML文件
graph.render("relation_word_graph.html")





# data是您的原始数据列表
new_data = [{k: v for k, v in d.items() if k in ['relation_word', 'relation_score', 'score_rank']} for d in data]


# 导入pyecharts库
from pyecharts import options as opts
from pyecharts.charts import Bar

# 定义你的数据
data =new_data

# 把数据转换为DataFrame对象
df = pd.DataFrame(data)



print(df)
# 把relation_score的类型转换为整数
df['relation_score'] = df['relation_score'].astype(float)
df['relation_score'] = df['relation_score'].multiply(10000)
df['relation_score'] = df['relation_score'].astype(int)
df['score_rank'] = df['score_rank'].astype(int)


base_score = None
for d in df:
    if d['score_rank'] == 1:
        base_score = d['relation_score']
        break


for d in df:
    d['relation_percent'] = d['relation_score'] / base_score * 100

import pandas as pd


base_score = None
for index, row in df.iterrows():
    if row['score_rank'] == 1:
        base_score = row['relation_score']
        break

print(base_score)

for index, row in df.iterrows():
    df.at[index, 'relation_percent'] = row['relation_score'] / base_score * 100

df['relation_percent'] = df['relation_percent'].astype(int)


df= df.drop(columns=["relation_score"])

df= df.drop(columns=["score_rank"])


# 创建一个Bar对象
bar = Bar()

# 添加X轴数据，用排名作为X轴
bar.add_xaxis(df['relation_percent'].tolist())

# 添加Y轴数据，用relation_word作为Y轴
bar.add_yaxis("关系词", df['relation_word'].tolist())

# 设置图表的全局选项
bar.set_global_opts(
    title_opts=opts.TitleOpts(title="数据统计"), # 设置图表的标题
    xaxis_opts=opts.AxisOpts(name="排名"), # 设置X轴的名称
    yaxis_opts=opts.AxisOpts(name="关系词"), # 设置Y轴的名称
    toolbox_opts=opts.ToolboxOpts(), # 设置工具箱选项
    tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}"), # 设置提示框的格式
)

# 设置图表的系列选项
bar.set_series_opts(
    label_opts=opts.LabelOpts(position="right"), # 设置标签的位置
    itemstyle_opts=opts.ItemStyleOpts(color="#00CD96"), # 设置条形图的颜色
)

# 把图表转换为水平条形图
bar.reversal_axis()

# 渲染图表
bar.render("bar.html")


bar = Bar()

# 添加x轴和y轴数据，注意要将DataFrame的列转换为列表
bar.add_xaxis(df['relation_word'].tolist())
bar.add_yaxis("关联词与百分比", df['relation_percent'].tolist())

# 设置全局配置项，例如标题，工具箱，标签等
bar.set_global_opts(
    title_opts={"text": "关联词与百分比"},
    toolbox_opts={"show": True},
    
)

# 调用reversal_axis方法，将x轴和y轴交换，实现水平的条形图
bar.reversal_axis()

# 生成HTML文件，或者在Jupyter Notebook中直接显示
bar.render("bar.html")




from pyecharts.charts import Bar, Page


# 初始化一个Page对象
page = Page()

# 设置每页显示的条数
page_size = 25

# 计算总页数
total_page = (len(df) - 1) // page_size + 1

# 循环遍历每一页的数据
for i in range(total_page):
    # 获取每一页的起始和结束索引
    start = i * page_size
    end = min((i + 1) * page_size, len(df))
    
    # 获取每一页的子DataFrame
    sub_df = df.iloc[start:end]
    
    # 初始化一个Bar对象
    bar = Bar()
    
    # 添加x轴和y轴数据，注意要将DataFrame的列转换为列表
    bar.add_xaxis(sub_df['relation_word'].tolist())
    bar.add_yaxis("关联词与百分比", sub_df['relation_percent'].tolist())
    
    # 设置全局配置项，例如标题，工具箱，标签等
    bar.set_global_opts(
        title_opts={"text": "关联词与百分比（第{}页）".format(i + 1)},
        toolbox_opts={"show": True},
        
    )
    
    # 调用reversal_axis方法，将x轴和y轴交换，实现水平的条形图
    bar.reversal_axis()
    
    # 将Bar对象添加到Page对象中
    page.add(bar)

# 生成HTML文件，或者在Jupyter Notebook中直接显示
page.render("page.html")