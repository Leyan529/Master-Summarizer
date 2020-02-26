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
			# "date": {"$type": "date"}
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
			# "date": {"$type": "date"}
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


# Health_personal_Care
class Health_personal_Care():
	def __init__(self):
		self.main_cat = 'Health & Personal Care'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			# "date": {"$type": "date"}
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

# CellPhones_Accessories
class CellPhones_Accessories():
	def __init__(self):
		self.main_cat = 'Cell Phones & Accessories'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			# "date": {"$type": "date"}
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


# Camera_Photo
class Camera_Photo():
	def __init__(self):
		self.main_cat = 'Camera & Photo'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			# "date": {"$type": "date"}
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


# GPS_Navigation
class GPS_Navigation():
	def __init__(self):
		self.main_cat = 'GPS & Navigation'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			# "date": {"$type": "date"}
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

# Musical Instruments
class Music_Instrum():
	def __init__(self):
		self.main_cat = 'Musical Instruments'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			# "date": {"$type": "date"}
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
			"vote": {"$gte": 1},
		}
		return key

# Software
class Software():
	def __init__(self):
		self.main_cat = 'Software'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			# "date": {"$type": "date"}
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
			"vote": {"$gte": 1},
		}
		return key

# Computers
class Computers():
	def __init__(self):
		self.main_cat = 'Computers'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			# "date": {"$type": "date"}
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
			"vote": {"$gte": 1},
		}
		return key

# Video Games
class Video_Games():
	def __init__(self):
		self.main_cat = 'Video Games'

	def getAttr(self):
		return self.main_cat

	def getProductKey(self):
		key = {
			"description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
			"main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
			# "date": {"$type": "date"}
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
			"vote": {"$gte": 1},
		}
		return key

'''
    "Musical Instruments", 236420
    "Software", 116519
    "Computers", 653450
    "Sports & Outdoors", 1376309
    "Health & Personal Care", 115155
    "Automotive",
    "All Beauty", 39221
    "Cell Phones & Accessories",
    "Office Products",
    "Baby", 43701
    "Pet Supplies", 677196
    "Video Games", 129331
    "GPS & Navigation", 6929
    "Books",
    "Luxury Beauty",
'''

'''
db.getCollection('new_reviews2').find(
    {"main_cat": {'$eq': 'Musical Instruments'}, "vote": {"$gte": 1}}
     ).count()
'''



'''
main_cat = CellPhones_Accessories().getAttr()
main_cat = Camera_Photo().getAttr()
main_cat = GPS_Navigation().getAttr()
main_cat = Music_Instrum().getAttr()
main_cat = Software().getAttr()
main_cat = Computers().getAttr()
main_cat = Video_Games().getAttr()
'''