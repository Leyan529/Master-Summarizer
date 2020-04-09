from glob import glob

loggerName = "Text-Summary"
vocab_size = 60000
# vocab_size = 300000 # bert

# word_emb_type = 'word2Vec' # glove , bert , gpt-2


data_type = 'Cameras_new8'
# data_type = 'Cameras_high_acc_3' # Cameras_high_acc_3  loss可降至1以內

keywords = "POS_FOP_keywords"

Data_path_ = '../Train-Data/'
Data_path = '../Train-Data/%s/'%(data_type)

word_emb_type = 'word2Vec' # glove , bert , gpt-2
word_emb_path = Data_path + "Embedding/%s/%s.300d.txt"%(word_emb_type,word_emb_type)
vocab_path = Data_path + 'Embedding/%s/word.vocab'%(word_emb_type)
xls_path = Data_path +"/pro_review.xlsx"




max_key_num = 8

# Hyperparameters
hidden_dim = 512
emb_dim = 300
# emb_dim = 768 # bert
# emb_dim = 1024 # transform-xl
batch_size = 8
max_enc_steps = 1000		#99% of the articles are within length 55 # 1000
# max_enc_steps = 500		# Bert constrain

max_dec_steps = 50		#99% of the titles are within length 15
beam_size = 16
min_dec_steps= 4
gound_truth_prob = 0.1  # 0.25 Probabilities indicating whether to use ground truth labels instead of previous decoded tokens


# lr = 0.00002 # 0.001
lr = 0.001 # 0.001
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12 # 学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率
max_iterations = 500000
max_epochs = 100
# ber_layer = 11

save_model_path = "model/saved_models"

# intra_encoder = False
# intra_decoder = False

intra_encoder = True
intra_decoder = True

key_attention = False
emb_grad = False

