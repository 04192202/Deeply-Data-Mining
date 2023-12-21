import re
import requests
import pandas as pd
import time
from tqdm import trange


SESSDATA = " "

cookie = " "
cookie += f";SESSDATA={SESSDATA}"
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "cookie": cookie,
}
def get_info(vid):
    url = f"https://api.bilibili.com/x/web-interface/view/detail?bvid={vid}"
    response = requests.get(url, headers=headers)
    response.encoding = "utf-8"
    if response.status_code != 200:
        print(f"请求失败，状态码：{response.status_code}")
        return None

    data = response.json()
    info = {
        "标题": data["data"]["View"]["title"],
        "总弹幕数": data["data"]["View"]["stat"]["danmaku"],
        "视频数量": data["data"]["View"]["videos"],
        "cid": [dic["cid"] for dic in data["data"]["View"]["pages"]]
    }
    if info["视频数量"] > 1:
        info["子标题"] = [dic["part"] for dic in data["data"]["View"]["pages"]]
    return info

def get_danmu(info, start, end):
    all_dms = []
    date_list = pd.date_range(start, end).strftime("%Y-%m-%d")
    for i, cid in enumerate(info["cid"]):
        dms = []
        for date in trange(len(date_list)):
            url = f"https://api.bilibili.com/x/v2/dm/web/history/seg.so?type=1&oid={cid}&date={date_list[date]}"
            response = requests.get(url, headers=headers)
            response.encoding = "utf-8"
            data = re.findall(r"[:](.*?)[@]", response.text)
            dms += [dm[1:] for dm in data]
            time.sleep(0)
        all_dms += dms
    return all_dms

if __name__ == "__main__":
    vid = input("输入视频编号: ")
    info = get_info(vid)
    if info:
        for key, value in info.items():
            print(f"{key}: {value}")

        start = input("输入弹幕开始时间（年-月-日）: ")
        end = input("输入弹幕结束时间（年-月-日）: ")
        danmu = get_danmu(info, start, end)
        with open("danmu.txt", "w", encoding="utf-8") as fout:
            for dm in danmu:
                fout.write(dm + "\n")
                print(dm)