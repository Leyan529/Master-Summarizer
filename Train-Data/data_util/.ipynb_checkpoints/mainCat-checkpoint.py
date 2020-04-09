import dateutil.parser as parser
from datetime import datetime


# All_Electronics
class All_Electronics():
	def __init__(self):
		self.main_cat = 'All Electronics'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			"date": {"$type": "date"}
			# "feature": {"$type": "array"}
		}
		return key

	def getReviewKey(self):
		# 增設條件if asin in [as1,as2,as3]
		# 增設條件 找出前三名
		# 預設查看前5個產品
		# Query review
		key = {
			"main_cat": {'$eq': self.main_cat},
			# 			'big_categories': self.category1,
			#         'small_categories': category2,
			# 			'$or': [
			# 				{ "small_categories":"DVD Players" },
			# 				{"small_categories":"Portable DVD Players"},
			# 				{"small_categories":"DVD Players & Recorders"},
			# 				{"small_categories":"HD DVD Players"}
			# 			],
			"vote": {"$gte": 1},
			# 			'reviewTime':{"$gte":self.cond_date,"$type":"date"}
		}
		return key


# Pet_Supplies
class Pet_Supplies():
	def __init__(self):
		self.main_cat = 'Pet Supplies'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			"date": {"$type": "date"}
			# "feature": {"$type": "array"}
		}
		return key

	def getReviewKey(self):
		# 增設條件if asin in [as1,as2,as3]
		# 增設條件 找出前三名
		# 預設查看前5個產品
		# Query review
		key = {
			"main_cat": {'$eq': self.main_cat},
			# 			'big_categories': self.category1,
			#         'small_categories': category2,
			# 			'$or': [
			# 				{ "small_categories":"DVD Players" },
			# 				{"small_categories":"Portable DVD Players"},
			# 				{"small_categories":"DVD Players & Recorders"},
			# 				{"small_categories":"HD DVD Players"}
			# 			],
			"vote": {"$gte": 1},
			# 			'reviewTime':{"$gte":self.cond_date,"$type":"date"}
		}
		return key


# Sports_Outdoors
class Sports_Outdoors():
	def __init__(self):
		self.main_cat = 'Sports & Outdoors'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			"date": {"$type": "date"}
			# "feature": {"$type": "array"}
		}
		return key

	def getReviewKey(self):
		# 增設條件if asin in [as1,as2,as3]
		# 增設條件 找出前三名
		# 預設查看前5個產品
		# Query review
		key = {
			"main_cat": {'$eq': self.main_cat},
			# 			'big_categories': self.category1,
			#         'small_categories': category2,
			# 			'$or': [
			# 				{ "small_categories":"DVD Players" },
			# 				{"small_categories":"Portable DVD Players"},
			# 				{"small_categories":"DVD Players & Recorders"},
			# 				{"small_categories":"HD DVD Players"}
			# 			],
			"vote": {"$gte": 1},
			# 			'reviewTime':{"$gte":self.cond_date,"$type":"date"}
		}
		return key