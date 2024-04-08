# -*- coding:utf-8 -*-
import requests
from lxml import etree
import pandas as pd
from fake_useragent import UserAgent
ua = UserAgent()

root_path = r'./'

data = pd.read_excel(root_path + 'data.xlsx')

nrows = data.shape[0]

with pd.ExcelWriter('static_output.xlsx') as writer:
    for i in range(0, nrows):
        output = pd.DataFrame(columns=["title", "href", "date"])

        url = data.iloc[i, 1]  # 爬取页面url地址
        code = requests.get(url).encoding
        ua = UserAgent()
        headers = {
            "User-Agent": ua.chrome
        }

        j = 0
        title = list()
        href = list()
        date = list()
        print('================================================================')
        print(data.iloc[i, 0])
        print('================================================================')
        print(url)
        # 根据url获取页面源码
        resp = requests.get(url, headers=headers).text
        resp = resp.encode(code).decode('utf-8')
        # 使用etree对htmL字符串解析
        html = etree.HTML(resp)

        # 获取元素
        this_title = html.xpath(data.iloc[i, 2])
        href1 = data.iloc[i, 3]
        href2 = html.xpath(data.iloc[i, 4])
        this_href = [href1 + x for x in href2]
        this_date = html.xpath(data.iloc[i, 5])

        title = title + this_title
        href = href + this_href
        date = date + this_date

        if pd.isnull(data.iloc[i, 6]):
            continue
        else:
            url = data.iloc[i, 6]
            page_index = str(data.iloc[i, 7]).split(',')
            if len(page_index) == 1:
                current_page = data.iloc[i, 6][int(page_index[0])]
            elif len(page_index) == 2:
                current_page = data.iloc[i, 6][int(page_index[0]): (int(page_index[1])+1)]
            else:
                print("索引输入错误！")
            delta = data.iloc[i, 8]
            current_page = int(current_page)

            this_title = [1]
            while len(this_title) != 0:
                print(url)
                resp = requests.get(url, headers=headers).text
                resp = resp.encode(code).decode('utf-8')
                # 使用etree对htmL字符串解析
                html = etree.HTML(resp)

                # 获取元素
                this_title = html.xpath(data.iloc[i, 2])
                href1 = data.iloc[i, 3]
                href2 = html.xpath(data.iloc[i, 4])
                this_href = [href1 + x for x in href2]
                this_date = html.xpath(data.iloc[i, 5])

                # 用于判断是否出现url中页码数超实际页码数时，还能读取的情况。
                # 极小概率会使页面爬取提前结束（标准：上一页的所有标题与本页所有标题完全一样）
                if title[-len(this_title):] == this_title:
                    print('内容重复退出。')
                    break

                title = title + this_title
                href = href + this_href
                date = date + this_date

                current_page = current_page + int(delta)
                url = data.iloc[i, 6][:int(page_index[0])] + str(current_page) + data.iloc[i, 6][(int(page_index[-1])+1):]

        output['title'] = title
        output['href'] = href
        output['date'] = date
        output.to_excel(writer, sheet_name=data.iloc[i, 0])

