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

relation_word_list = provided_data.get('relation_word_list', [])


data_=  (relation_word_list)
data_




import networkx as nx
import matplotlib.pyplot as plt


data = data_

G = nx.DiGraph()

# 添加节点和边
for _, item in data.iterrows():
    G.add_node(item['relation_word'], size=item['search_index'])
    G.add_edge('婚 姻', item['relation_word'], weight=item['compos_index'])

# 绘制节点
pos = nx.spring_layout(G)
sizes = [data['search_index'][node] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='skyblue', alpha=0.8)

# 绘制边
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='gray')

# 添加标签
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

# 显示图形
plt.title('关联词图')
plt.show()

































