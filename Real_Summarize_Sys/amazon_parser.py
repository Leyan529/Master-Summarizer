#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import csv
import sys
import os
import fnmatch
import re
from bs4 import BeautifulSoup

if sys.version_info[0] >= 3:
    import html


def get_review_filesnames(input_dir):
    for root, dirnames, filenames in os.walk(input_dir):
        for filename in fnmatch.filter(filenames, '*.html'):
            yield os.path.join(root, filename)


idre = re.compile('product\-reviews/([A-Z0-9]+)/ref\=cm_cr_arp_d_hist', re.MULTILINE | re.S)
contentre = re.compile(
    'cm_cr-review_list.*?>(.*?)(?:askReviewsPageAskWidget|a-form-actions a-spacing-top-extra-large|/html)',
    re.MULTILINE | re.S)
blockre = re.compile('a-section review\">(.*?)report-abuse-link', re.MULTILINE | re.S)
ratingre = re.compile('star-(.) review-rating', re.MULTILINE | re.S)
titlere = re.compile('review-title.*?>(.*?)</a>', re.MULTILINE | re.S)
datere = re.compile('review-date">(.*?)</span>', re.MULTILINE | re.S)
reviewre = re.compile('base review-text">(.*?)</span', re.MULTILINE | re.S)
userre = re.compile('profile\/(.*?)["/].*?\<\/div\>.*?\<\/div\>.', re.MULTILINE | re.S)
helpfulre = re.compile('review-votes.*?([0-9]+).*?([0-9]+)', re.MULTILINE | re.S)


# def main():
# sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)
parser = argparse.ArgumentParser(
    description='Amazon review parser')
# parser.add_argument('-d', '--dir', default='amazonreviews\com\B019U00D7K', help='Directory with the data for parsing', required=False)
# parser.add_argument('-o', '--outfile', default='B019U00D7K.csv', help='Output file path for saving the reviews in csv format', required=False)

parser.add_argument('--ids', type=list, default=[
    # 'B019U00D7K',
    # 'B07VJRZ62R',
    'B07T6Y2HG7'
    ], help='Product IDs for which to download reviews', required=False)
    
args = parser.parse_args()

reviews = dict()

for id_ in args.ids:
    with codecs.open('%s.csv'%(id_), 'w', encoding='utf-8', errors='ignore') as out:
        writer = csv.writer(out, lineterminator='\n')
        for filepath in get_review_filesnames('amazonreviews\com\%s' % id_):
            with codecs.open(filepath, mode='r', encoding='utf-8', errors='ignore') as file:
                htmlpage = file.read()

            soup = BeautifulSoup(htmlpage,'lxml')
            blocks = soup.findAll("div", {"data-hook" : "review"})

            for block in blocks:
                try:
                    summary = block.findAll("a", {"class" : "a-link-normal"})[1].text.strip()
                    reviewtext = block.find("div", {"class" : "a-spacing-small"}).text.strip()
                    rating = float(block.findAll("a", {"class" : "a-link-normal"})[0].text.strip().split(" ")[0])
                    date = block.find("span", {"class" : "review-date"}).text.strip()
                    date = " ".join(date.split(" ")[6:])
                    comments = block.find("div", {"class" : "review-comments"}).text.strip()
                except:
                    continue

                if rating >= 4:
                    binaryrating = 'positive'
                else:
                    binaryrating = 'negative'
                review_row = [date, summary, reviewtext, rating, binaryrating]
                writer.writerow(review_row)


# if __name__ == '__main__':
#     main()