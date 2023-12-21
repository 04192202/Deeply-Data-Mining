
import json

# 读取JSON文件
with open('response_1.json', 'r', encoding='utf-8') as file:
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
age_data = provided_data['data'][0]['label_list']
gender_data = provided_data['data'][1]['label_list']
province_data = provided_data['data'][2]['label_list']
city_data = provided_data['data'][3]['label_list']
label_data = provided_data['data'][4]['label_list']
interest_data = provided_data['data'][5]['label_list']




#以province_data为例
data_pp= pd.DataFrame(province_data)

data_pp= data_pp.drop(columns=["label_id"])

data_pp= data_pp.sort_values("value", ascending=False)
total = data_pp["value"].sum()
data_pp["分布占比"] = data_pp["value"] / total *100
data_pp= data_pp.drop(columns=["value"])

data_pp

# 定义一个函数，给每个值添加省字
def add_province(x):
    return x + "省"

# 使用apply()函数，把add_province()函数应用到data_pp["name_zh"]列中的每个值
data_pp["name_zh"] = data_pp["name_zh"].apply(add_province)

name_dict = {"北京省":"北京市", "天津省":"天津市", "上海省":"上海市", "重庆省":"重庆市", 
             "内蒙古省":"内蒙古自治区", "广西省":"广西壮族自治区", "西藏省":"西藏自治区", 
             "宁夏省":"宁夏回族自治区", "新疆省":"新疆维吾尔自治区","香港省":"香港特别行政区","澳门省":"澳门特别行政区"}

data_pp["name_zh"] = data_pp["name_zh"].replace(name_dict)


from pyecharts.charts import Map, Grid,Bar
from pyecharts import options as opts



m = Map()

m.add("TGI指数", [list(z) for z in zip(data_pp['name_zh'], data_pp['tgi'])])
m.add("分布占比", [list(z) for z in zip(data_pp['name_zh'], data_pp['分布占比'])])

m.set_global_opts(
    
    title_opts=opts.TitleOpts(title="全国TGI指数地图"),
    visualmap_opts=opts.VisualMapOpts(max_=120, min_=90, is_piecewise=True, pieces=[{"min": 110, "color": "#003366"}, {"min": 100, "max": 110, "color": "#156ACF"}, {"min": 90, "max": 100, "color": "#5CACEE"}]),
    toolbox_opts=opts.ToolboxOpts(is_show=True),
    tooltip_opts=opts.TooltipOpts(is_show=True, trigger="item", formatter="{b}<br/>TGI指数: {c0}<br/>")
   
)
# 生成html文件
m.render('TGI_map.html')