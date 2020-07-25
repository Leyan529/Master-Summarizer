# user input

import os
import sys
import codecs
import argparse
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

if sys.version_info[0] >= 3:
    import urllib
    import urllib.request as request
    import urllib.error as urlerror
else:
    import urllib2 as request
    import urllib2 as urlerror
import socket
from contextlib import closing
from time import sleep
import re

from my_parser import parse_page
from image import download_img

import pandas as pd

def download_page(url, referer, maxretries, timeout, pause):
    tries = 0
    htmlpage = None
    while tries < maxretries and htmlpage is None:
        try:
            code = 404
            req = request.Request(url)
            req.add_header('Referer', referer)
            ua=UserAgent()
            req.add_header('User-agent',ua.random)
            
            with closing(request.urlopen(req, timeout=timeout)) as f:
                code = f.getcode()
                htmlpage = f.read()
                sleep(pause)
        except (urlerror.URLError, socket.timeout, socket.error):
            tries += 1
    if htmlpage:
        return htmlpage.decode('utf-8'), code
    else:
        return None, code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', help='Domain from which to download the reviews. Default: com',
                        required=False,
                        type=str, default='com')
    parser.add_argument('-f', '--force', help='Force download even if already successfully downloaded', required=False,
                        action='store_true')
    parser.add_argument(
        '-r', '--maxretries', help='Max retries to download a file. Default: 3',
        required=False, type=int, default=3)
    parser.add_argument(
        '-t', '--timeout', help='Timeout in seconds for http connections. Default: 180',
        required=False, type=int, default=180)
    parser.add_argument(
        '-p', '--pause', help='Seconds to wait between http requests. Default: 1', required=False, default=5,
        type=float)
    parser.add_argument(
        '-m', '--maxreviews', help='Maximum number of reviews per item to download. Default:unlimited',
        required=False,
        type=int, default=-1)
 
    # B00CJICDSS (TOUGH-GRID 750磅傘繩/降落傘繩 - 美國軍方使用的正版 Mil Spec IV 750磅傘繩 (MIl-C-5040-H) - 100% 尼龍 - 美國製造) # page 426 1001
    # B003VVP0KU (Prevue Hendryx 旅行鳥籠 1305 白色，50.8 公分 x 30.5 公分 x 38.1 公分) # page 48 186  
    # ---------------------------------------------------------------------------------------------------------------------------
    # 電器類 
    # B004NBZB2Y (o) Texas Instruments - Ti-36X 太陽能科學計算機  # page 298 769
    # ---------------------------------------------------------------------------------------------------------------------------
    # 健康與個人照護類 
    # B00DVPS4IQ Marijuana (THC) 單片式藥物測試組 # page 250 431
    # ---------------------------------------------------------------------------------------------------------------------------
    # 相機與照片類 
    # B01HGM33HG AKASO EK7000 4K WiFi 運動攝影機 # page 67 247
    # ---------------------------------------------------------------------------------------------------------------------------
    # 手機及配件類     
    # B07YN7CMXS (Xiaomi Redmi Note 8 Pro) 128GB, 6GB RAM 6.53" LTE GSM 64MP Factory Unlocked Smartphone  
    # ---------------------------------------------------------------------------------------------------------------------------
    # B07F981R8M (TCL Smart LED TV 4W 白色 LED燈), 32S327) # page 152 411    
    # B076H3SRXG Wyze Cam 網路攝影機 1080p HD 室內 無線 智慧居家 夜視攝影機 # page 377 1002
    # B07145GM4B PHILIPS 飛利浦 Norelco Multi Groomer MG7750/49 23 件組 多功能修容器 # page 309 748
    # B07P6Y7954 Apple iPhone XR Fully Unlock(Renewed) 黑色 # page 72 168

    
    
    parser.add_argument('--id', type=str, default= 'B01HGM33HG', choices=['B071CV8CG2','B019U00D7K'],
                        help='Product IDs for which to download reviews', required=False)
    parser.add_argument('-c', '--captcha', help='Retry on captcha pages until captcha is not asked. Default: skip', required=False,
                    action='store_true')
    args = parser.parse_args()
    '''
    cd D:\WorkSpace\Real_sum_sys
    d:
    activate tensorflow
    python my_crawl_data.py
    '''

    print(args.id)
    # ASID = args.id
    id_ = args.id
    if not os.path.exists(id_):
        os.makedirs(id_)

    urlPart1 = "http://www.amazon.com/product-reviews/"
    urlPart2 = "/?ie=UTF8&showViewpoints=0&pageNumber="
    urlPart3 = "&sortBy=bySubmissionDateDescending"
    urlPart4 = "&rh=p_72%3A2661618011&dc&qid=1592571821&rnid=2661617011&ref=sr_nr_p_72_1"
    product_url = "https://www.amazon.com/dp/%s/ref=cm_cr_arp_d_product_top?ie=UTF8"%(id_)

    counterre = re.compile('cm_cr_arp_d_paging_btm_([0-9]+)')
    robotre = re.compile('images-amazon\.com/captcha/')

    referer = urlPart1 + str(id_) + urlPart2 + "1" + urlPart3

    page = 1
    lastPage = 1

    total_review_row = []
    total_df = None

    # basepath = args.out + os.sep + args.domain
    # img_path = basepath + os.sep + id_
    download_img(None, id_)
    htmlpage, code = download_page(product_url, referer, args.maxretries, args.timeout, args.pause)
    soup = BeautifulSoup(htmlpage,'lxml')
    prod_dict = {}
    prod_dict["productTitle"] = soup.find("span", {"id" : "productTitle"}).text.strip()
    prod_dict["acrCustomerReviewText"] = soup.find("span", {"id" : "acrCustomerReviewText"}).text.strip()
    prod_dict["feature-bullets"] = soup.find("div", {"id" : "feature-bullets"}).text.strip()
    TAG_RE = re.compile(r'<[^>]+>')
    with open('%s/info.txt'%(id_), 'w', encoding='utf-8') as f:
        for k, v in prod_dict.items():
            v = re.sub(' +',' ',v) # Removing extra spaces
            v = TAG_RE.sub('', v)
            v = re.sub('\n+','\n',v) # Removing extra spaces
            f.write(k + ': ' + v + '\n')

    while page <= lastPage:
        # if not page == 1 and not args.force and os.path.exists(basepath + os.sep + id_ + os.sep + id_ + '_' + str(
        #         page) + '.html'):
        #     print('Already got page ' + str(page) + ' for product ' + id_)
        #     page += 1
        #     continue
        
        url = urlPart1 + str(id_) + urlPart2 + str(page) + urlPart3
        # print(url)
        htmlpage, code = download_page(url, referer, args.maxretries, args.timeout, args.pause)

        if htmlpage is None or code != 200:
            if code == 503:
                page -= 1
                args.pause += 2
                print('(' + str(code) + ') Retrying downloading the URL: ' + url)
            else:
                print('(' + str(code) + ') Done downloading the URL: ' + url)
                break
        else:
            print('Got page ' + str(page) + ' out of ' + str(lastPage) + ' for product ' + id_ + ' timeout=' + str(
                args.pause))
            if robotre.search(htmlpage):
                print('ROBOT! timeout=' + str(args.pause))
                if args.captcha or page == 1:
                    args.pause *= 2
                    continue
                else:
                    args.pause += 2
            for match in counterre.findall(htmlpage):
                try:
                    value = int(match)
                    if value > lastPage:
                        lastPage = value
                except:
                    pass
            
        df = parse_page(id_, htmlpage)
        if page == 1:
            total_df = df
        else:
            total_df = pd.concat([total_df, df])
        

        if len(total_df[total_df["binaryrating"] == "negative"]) >= 5:
            posit_df = total_df[total_df["binaryrating"] == "positive"]
            negat_df = total_df[total_df["binaryrating"] == "negative"]

            posit_df = posit_df.sort_values(by=['date'], ascending = False)
            negat_df = negat_df.sort_values(by=['date'], ascending = False)

            # posit_df = posit_df[:5]
            # negat_df = negat_df[:5]
            
            # list(negat_df['reviewtext']); list(negat_df['lemm_reviewtext'])
            if len(total_df) % 1 == 0:
                print('save data %s'%(len(total_df)))
                total_df.to_excel('%s/total.xlsx'%(id_))
                posit_df.to_excel('%s/positive.xlsx'%(id_))
                negat_df.to_excel('%s/negative.xlsx'%(id_))

            # break   
        page += 1  
        if len(total_df)>1000: 
            print('save %s data finished %s'%(id_, len(total_df)))
            break

    # print(posit_df[['date','rating']].head())
    # print(negat_df[['date','rating']].head())
    
