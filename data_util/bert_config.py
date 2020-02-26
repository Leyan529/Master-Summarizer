from data_util.product import *
from glob import glob

loggerName = "Text-Summary"
vocab_size = 50000
# vocab_size = 300000 # bert


# 內置在Text Review FOP裡的 data
# data_type = 'category' # or main_cat
# # data_type = 'main_cat' # or main_cat
# word_emb_type = 'word2Vec' # glove , bert , gpt-2
# word_emb_path = "Embedding/%s/%s/%s.300d.txt"%(data_type,word_emb_type,word_emb_type)
# vocab_path = 'Embedding/category/%s/word.vocab'%(word_emb_type)

# train_data_path = 	"bin/%s/chunked/train/train_*"%(data_type)
# valid_data_path = 	"bin/%s/chunked/valid/valid_*"%(data_type)
# test_data_path = 	"bin/%s/chunked/test/test_*"%(data_type)

data_type = 'Cameras' # Cameras makeRecord-bert2 (新移除所有怪異字母 + alphabet,調參使的accuracy可降至1以內)
# data_type = 'Cameras_no_cheat' # Cameras makeRecord-bert2 (新移除所有怪異字母 + alphabet,調參使的accuracy可降至1以內)
# 2_2 > 3_1 > 2_1
# data_type = 'Cameras3_2'

# data_type = 'Cameras_high_acc' # Cameras_high_acc makeRecord-bert2 (調參使的accuracy可降至1以內)
# data_type = 'Cameras_high_acc_3' # Cameras_high_acc_3  loss可降至1以內
Data_path_ = '/home/eagleuser/Users/leyan/Train-Data/'
Data_path = '/home/eagleuser/Users/leyan/Train-Data/%s/'%(data_type)

word_emb_type = 'word2Vec' # glove , bert , gpt-2
word_emb_path = Data_path + "Embedding/%s/%s.300d.txt"%(word_emb_type,word_emb_type)
vocab_path = Data_path + 'Embedding/%s/word.vocab'%(word_emb_type)

train_data_path = Data_path + "bin/chunked/train/train_*"
valid_data_path = Data_path + "bin/chunked/valid/valid_*"
test_data_path = Data_path + "bin/chunked/test/test_*"
bin_info = Data_path +"bin/bin-info.txt"


keywords = "FOP_keywords"
# keywords = "TextRank_keywords"
max_key_num = 8

# Hyperparameters
hidden_dim = 512
emb_dim = 300
# emb_dim = 768 # bert
# emb_dim = 1024 # transform-xl
batch_size = 16
max_enc_steps = 1000		#99% of the articles are within length 55 # 1000
# max_enc_steps = 512		# Bert constrain

max_dec_steps = 60		#99% of the titles are within length 15
beam_size = 16
min_dec_steps= 5
gound_truth_prob = 0.25  # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens


# lr = 0.00002 # 0.001
lr = 0.0001 # 0.001
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
max_iterations = 500000
max_epochs = 100


save_model_path = "model/saved_models"

# intra_encoder = False
# intra_decoder = False

intra_encoder = True
intra_decoder = True


# #--------------# Train mode
# train_mle = 'yes'
# train_rl = 'no'
# mle_weight = 1.0
# load_model = None
# new_lr = None


