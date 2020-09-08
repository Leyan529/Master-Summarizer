# Master-thesis
https://hackmd.io/hfJZN4wyQ4uoHqL3nz0uUg
## Abstractive Summary model + Self-Critical Sequence Training
![](https://i.imgur.com/0wRbOpx.png)

#### 1. PG-intra
#### 2. PG-intra-RL
#### 3. PG-key
#### 4. PG-key-RL
#### 5. PG-intra-key
#### 6. PG-intra-key-RL


![](https://i.imgur.com/rThJOuj.png)
![](https://i.imgur.com/JIBHiRq.png)
![](https://i.imgur.com/HPByDHB.png)


## Amazon Product Review Test
#### 以人工判讀之參考摘要進行分析
![](https://i.imgur.com/546VYyk.png)
#### 以破題法取英文第一句判讀之摘要進行分析
![](https://i.imgur.com/qIA9vve.png)


## Structure
### Train-Data
    1. Create the Training csv (pro_review.xlsx)
    2. Create the Corpus、CBOW pre-train weight & Vocab
    
### Summarize_parallel
    1. Main Traing Code (Finally use)
    2. Multiple GPU Traing flow

### Real_Summarize_Crawl
    Crawl Product review by asin
    my_crawl_data.py
    
### Lingu_Feat
    Evaluate FOP feature
