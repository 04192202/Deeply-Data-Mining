import json
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Line

# 读取JSON文件
with open('response.json', 'r', encoding='utf-8') as file:
    json_data = file.read()



# 解析JSON数据
data = json.loads(json_data)
data_dict = json.loads(data['data'])

provided_data=data_dict
marry_time_data = provided_data['hot_list'][0]['hot_list']
divorce_time_data = provided_data['hot_list'][1]['hot_list']




spread_data=marry_time_data

dates = [entry['datetime'] for entry in spread_data]
indices = [int(entry['index']) for entry in spread_data]


marry_time_2022_data = [(date, index) for date, index in zip(dates, indices) if date[:4] == '2022']
marry_time_2023_data = [(date, index) for date, index in zip(dates, indices) if date[:4] == '2023']

marry_time=[(date, index) for date, index in zip(dates, indices) ]


spread_data=divorce_time_data

divorce_time_data_dates = [entry['datetime'] for entry in spread_data]
divorce_time_data_indices = [int(entry['index']) for entry in spread_data]

divorce_time_data_2022_data = [(date, index) for date, index in zip(divorce_time_data_dates, divorce_time_data_indices) if date[:4] == '2022']
divorce_time_data_2023_data = [(date, index) for date, index in zip(divorce_time_data_dates, divorce_time_data_indices) if date[:4] == '2023']

divorce_time = [(date, index) for date, index in zip(divorce_time_data_dates, divorce_time_data_indices) ]


from pyecharts.charts import Line, Page,Grid,Timeline
from pyecharts import options as opts
from pyecharts.faker import Faker



chart = (
    Line()
    .add_xaxis([entry[0] for entry in marry_time])
    .add_yaxis("marry Index - ", [entry[1] for entry in marry_time], is_smooth=True)
    .add_yaxis("divorce Index - ", [entry[1] for entry in divorce_time], is_smooth=True)
    .set_global_opts(
        title_opts=opts.TitleOpts(title="marry vs divorce Index Comparison ", pos_top="5%"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45), name_gap=25),
       datazoom_opts=[opts.DataZoomOpts(pos_bottom="10%"), opts.DataZoomOpts(type_="inside")],
    )
)





grid_chart = (
    Grid()
    .add(chart, grid_opts=opts.GridOpts(pos_top="10%", pos_bottom="55%",height="80%", width="100%"))
)

# Render the grid to an HTML file
grid_chart.render("comparison_combined_chart.html")

