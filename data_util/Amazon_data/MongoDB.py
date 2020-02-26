from pymongo import MongoClient
import pymongo


class MongoDB:
	def __init__(self):
		host = '140.116.39.2'
		# host = 'localhost'
		port = 27017
		self.user_name = 'root'
		self.user_pwd = '1234'
		self.user_db = 'admin'
		mechanism = 'SCRAM-SHA-1'

		uri = "mongodb://{username}:{password}@{host}:{port}/{user_db}?authMechanism=SCRAM-SHA-1".format(
			username=self.user_name,
			password=self.user_pwd,
			host=host,
			port=port,
			user_db=self.user_db)

		print("conn_mongo -- uri: " + uri)
		self.mongo_client = MongoClient(uri)

	def conn_db(self, db_name):
		print("Auth : ", self.mongo_client[db_name].authenticate(self.user_name,
																 self.user_pwd,
																 self.user_db,
																 mechanism='SCRAM-SHA-1'))
		mongo_db = self.mongo_client[db_name]
		print("Connect to db : %s " % (db_name))
		self.mongo_db = mongo_db
		return mongo_db


	def make_index(self, db_col='Prod_Review',group_index = ["asin"]):
		mongo_coll = self.mongo_db[db_col]
		# mongo_coll.create_index([('asin', pymongo.TEXT),
		# 						 ('categories', pymongo.TEXT),
		# 						 ('main_helpful', pymongo.TEXT),
		# 						 ('review_token_len', pymongo.TEXT),
		# 						 ('summary_token_len', pymongo.TEXT)])  # 对名为field的项建立文档索引

		# group_index = [(index, pymongo.TEXT) for index in group_index]
		# print(str(group_index)
		# mongo_coll.create_index(group_index)

		mongo_coll.create_index([('asin', pymongo.TEXT),
								 ('date', pymongo.TEXT),
								 ('category1', pymongo.TEXT),
								 ('category2', pymongo.TEXT)])  # 对名为field的项建立文档索引

	def searchInDB(self, key, db_col='Prod_Review',ret_key={},limit=False):
		mongo_coll = self.mongo_db[db_col]
		if len(ret_key.items()) != 0:
			if limit:
				cursor = mongo_coll.find(key, ret_key, no_cursor_timeout=True).limit(limit)  # 此處須注意，其回傳的並不是資料本身，你必須在迴圈中逐一讀出來的過程中，它才真的會去資料庫把資料撈出來給你。
			else:
				cursor = mongo_coll.find(key, ret_key, no_cursor_timeout=True)  # 此處須注意，其回傳的並不是資料本身，你必須在迴圈中逐一讀出來的過程中，它才真的會去資料庫把資料撈出來給你。
		else:
			if limit:
				cursor = mongo_coll.find(key, no_cursor_timeout=True).limit(limit)  # 此處須注意，其回傳的並不是資料本身，你必須在迴圈中逐一讀出來的過程中，它才真的會去資料庫把資料撈出來給你。
			else:
				cursor = mongo_coll.find(key, no_cursor_timeout=True)  # 此處須注意，其回傳的並不是資料本身，你必須在迴圈中逐一讀出來的過程中，它才真的會去資料庫把資料撈出來給你。

		return cursor

	def insert(self, data, db_col='Prod_Review'):
		try:
			mongo_coll = self.mongo_db[db_col]
			mongo_coll.insert(data)
		except Exception as e:
			print(data)
			print("Insert Error")

	def searchInDBCount(self, key, db_col='Prod_Review',limit = False):
		mongo_coll = self.mongo_db[db_col]
		if limit:
			cursor = mongo_coll.find(key,no_cursor_timeout=True).limit(limit) # 此處須注意，其回傳的並不是資料本身，你必須在迴圈中逐一讀出來的過程中，它才真的會去資料庫把資料撈出來給你。
		else:
			cursor = mongo_coll.find(key, no_cursor_timeout=True)  # 此處須注意，其回傳的並不是資料本身，你必須在迴圈中逐一讀出來的過程中，它才真的會去資料庫把資料撈出來給你。
		print("Search key {} in DB :{} Documents".format(key, cursor.count()))

	def searchMainCategeory(self, key, db_col='Prod_Review'):
		mongo_coll = self.mongo_db[db_col]
		main_cat = mongo_coll.find(key,no_cursor_timeout=True).distinct("categories1")
		return main_cat[0]


	def count(self, key, db_col='Prod_Review'):
		mongo_coll = self.mongo_db[db_col]
		print("Toal {} douments in DB".format(mongo_coll.count()))