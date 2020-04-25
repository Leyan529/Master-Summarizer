from glob import glob

loggerName = "Text-Summary"
vocab_size = 60000
# vocab_size = 300000 # bert

# word_emb_type = 'word2Vec' # glove , bert , gpt-2


# data_type = 'Cameras_new8'
# data_type = 'Cameras_high_acc_3' # Cameras_high_acc_3  loss可降至1以內
data_type = 'Cameras'
# data_type = 'Mix6_mainCat'

keywords = "POS_FOP_keywords"

Data_path_ = '../Train-Data/'
Data_path = '../Train-Data/%s/'%(data_type)

word_emb_type = 'word2Vec' # glove , bert , gpt-2
word_emb_path = Data_path + "Embedding/%s/%s.300d.txt"%(word_emb_type,word_emb_type)
vocab_path = Data_path + 'Embedding/%s/word.vocab'%(word_emb_type)
xls_path = Data_path +"/pro_review.xlsx"
save_model_path = "model/saved_models"
emb_grad = False
max_key_num = 8
eps = 1e-12 # 学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率
coverage = True



