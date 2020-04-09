import dateutil.parser as parser
from datetime import datetime
# DVD Players
class DVD_Player():
	def __init__(self):
		self.main_cat = 'All Electronics'
		self.category1 = 'Electronics'
		self.category2 ='DVD Players'
# 		self.db_col='new_Product2'
		# self.cond_date = '2015-01-01'
		self.cond_date = datetime(1900, 1, 1, 0, 0, 0)
		
	def getAttr(self):
		return self.main_cat,self.category1,self.category2,self.cond_date
		
	def getProductKey(self):
		key = {
        "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
#         "main_cat": {'$eq': main_cat, "$type": "string", "$exists": True},
        "category1": {'$eq': self.category1, "$type": "string", "$exists": True},
#         "category2": {'$eq': category2, "$type": "string", "$exists": True},
        '$or': [
            { "category2":"DVD Players" },
            {"category2":"Portable DVD Players"},
            {"category2":"DVD Players & Recorders"},
            {"category2":"HD DVD Players"}  
        ],
        "title": {"$ne": "nan", "$exists": True, "$type": "string"},
        'salesRank': {"$exists": True,"$ne": ""},
#         "date":{"$gte":cond_date,"$type":"date"}
        "date":{"$type":"date"}
		}
		return key
	
	def getReviewKey(self):
		# 增設條件if asin in [as1,as2,as3]
		# 增設條件 找出前三名
		# 預設查看前5個產品
		# Query review
		key = {
	#         "main_cat": {'$eq': main_cat},
			'big_categories': self.category1,
	#         'small_categories': category2,
			'$or': [
				{ "small_categories":"DVD Players" },
				{"small_categories":"Portable DVD Players"},
				{"small_categories":"DVD Players & Recorders"},
				{"small_categories":"HD DVD Players"}  
			],
			"vote": {"$gte": 1},
			'reviewTime':{"$gte":self.cond_date,"$type":"date"}
		}
		return key
	
# Cameras
class Cameras():
	def __init__(self):
		self.main_cat = 'All Electronics'
		self.category1 = 'Electronics'
		self.category2 ='Cameras'
# 		self.db_col='new_Product2'
		# self.cond_date = '2015-01-01'
		self.cond_date = datetime(1900, 1, 1, 0, 0, 0)
		
	def getAttr(self):
		return self.main_cat,self.category1,self.category2,self.cond_date
		
	def getProductKey(self):
		key = {
        "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
#         "main_cat": {'$eq': main_cat, "$type": "string", "$exists": True},
        "category1": {'$eq': self.category1, "$type": "string", "$exists": True},
#         "category2": {'$eq': category2, "$type": "string", "$exists": True},
        '$or': [
                { "category2":"Dome Cameras"},
        { "category2":"Camera & Photo"},
        { "category2":"Camera Cases"},
        { "category2": "Professional Video Cameras"},
         { "category2":"On-Dash Cameras"},
         { "category2":"APS Cameras"},
         { "category2":"Point & Shoot Film Cameras"},
         { "category2":"Point & Shoot Digital Cameras"},
         { "category2":"Digital Cameras"},
         { "category2":"SLR Cameras" },
         { "category2":"Single-Use Cameras"},
          { "category2":"Film Cameras"},
          { "category2":"Bullet Cameras"},
          { "category2":"Surveillance Cameras"},
          { "category2":"Hidden Cameras"},
          { "category2":"Instant Cameras"},
          { "category2":"DSLR Cameras"},
          { "category2":"Cameras"},
          { "category2":"Simulated Cameras"},
          { "category2":"Medium & Large-Format Cameras"},
          { "category2":"Rangefinder Cameras"},
          { "category2":"Vehicle Backup Cameras"},
          { "category2":"Compatible with SLR cameras, digital cameras, 35mm cameras, and mini digital video cameras"},
          { "category2":"Specialty Film Cameras"},
          { "category2":"Sports & Action Video Cameras"},
          { "category2":"Mirrorless Cameras"},
          { "category2":"Body Mounted Cameras"}  ],
        "title": {"$ne": "nan", "$exists": True, "$type": "string"},
        'salesRank': {"$exists": True,"$ne": ""},
#         "date":{"$gte":cond_date,"$type":"date"}
        "date":{"$type":"date"}
		}
		return key
	
	def getReviewKey(self):
		# 增設條件if asin in [as1,as2,as3]
		# 增設條件 找出前三名
		# 預設查看前5個產品
		# Query review
		key = {
#         "main_cat": {'$eq': main_cat},
        'big_categories': self.category1,
#         'small_categories': category2,
        '$or': [
                { "small_categories":"Dome Cameras"},
        { "small_categories":"Camera & Photo"},
        { "small_categories":"Camera Cases"},
        { "small_categories": "Professional Video Cameras"},
         { "small_categories":"On-Dash Cameras"},
         { "small_categories":"APS Cameras"},
         { "small_categories":"Point & Shoot Film Cameras"},
         { "small_categories":"Point & Shoot Digital Cameras"},
         { "small_categories":"Digital Cameras"},
         { "small_categories":"SLR Cameras" },
         { "small_categories":"Single-Use Cameras"},
          { "small_categories":"Film Cameras"},
          { "small_categories":"Bullet Cameras"},
          { "small_categories":"Surveillance Cameras"},
          { "small_categories":"Hidden Cameras"},
          { "small_categories":"Instant Cameras"},
          { "small_categories":"DSLR Cameras"},
          { "small_categories":"Cameras"},
          { "small_categories":"Simulated Cameras"},
          { "small_categories":"Medium & Large-Format Cameras"},
          { "small_categories":"Rangefinder Cameras"},
          { "small_categories":"Vehicle Backup Cameras"},
          { "small_categories":"Compatible with SLR cameras, digital cameras, 35mm cameras, and mini digital video cameras"},
          { "small_categories":"Specialty Film Cameras"},
          { "small_categories":"Sports & Action Video Cameras"},
          { "small_categories":"Mirrorless Cameras"},
          { "small_categories":"Body Mounted Cameras"}  ],
        "vote": {"$gte": 1},
        'reviewTime':{"$gte":self.cond_date,"$type":"date"}
    }
		return key

# Cell Phones
class Cell_Phones():
	def __init__(self):
		self.main_cat = 'All Electronics'
		self.category1 = 'Cell Phones & Accessories'
		self.category2 ='Cell Phones' 
# 		self.db_col='new_Product2'
		# self.cond_date = '2015-01-01'
		self.cond_date = datetime(1900, 1, 1, 0, 0, 0)
		
	def getAttr(self):
		return self.main_cat,self.category1,self.category2,self.cond_date
		
	def getProductKey(self):
		key = {
        "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
#         "main_cat": {'$eq': main_cat, "$type": "string", "$exists": True},
        "category1": {'$eq': self.category1, "$type": "string", "$exists": True},
#         "category2": {'$eq': self.category2, "$type": "string", "$exists": True},
        '$or': [
                { "category2":"Cell Phones" },
        ],
#         "title": {"$ne": "nan", "$exists": True, "$type": "string"},
#         'salesRank': {"$exists": True,"$ne": ""},
#         "date":{"$gte":cond_date,"$type":"date"}
#         "date":{"$type":"date"}
    } 
		return key
	
	def getReviewKey(self):
		# 增設條件if asin in [as1,as2,as3]
		# 增設條件 找出前三名
		# 預設查看前5個產品
		# Query review
		key = {
#         "main_cat": {'$eq': main_cat},
        'big_categories': self.category1,
#         'small_categories': self.category2,
        '$or': [
                { "small_categories":"Cell Phones" }
        ],
        "vote": {"$gte": 1},
#         'reviewTime':{"$gte":self.cond_date,"$type":"date"}
    }
		return key
	
# GPS
class GPS():
	def __init__(self):
		self.main_cat = 'GPS & Navigation'
		self.category1 = 'Electronics'
		self.category2 ='GPS'  
# 		self.db_col='new_Product2'
		# self.cond_date = '2015-01-01'
		self.cond_date = datetime(1900, 1, 1, 0, 0, 0)
		
	def getAttr(self):
		return self.main_cat,self.category1,self.category2,self.cond_date
		
	def getProductKey(self):
		key = {
        "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
        "main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
        "category1": {'$eq': self.category1, "$type": "string", "$exists": True},
#         "category2": {'$eq': category2, "$type": "string", "$exists": True},
#         '$or': [
#                 { "category2":"Cell Phones" },
#                 {"category2":"Phones & tablets, Mobile phones"}
#         ],
        "title": {"$ne": "nan", "$exists": True, "$type": "string"},
        'salesRank': {"$exists": True,"$ne": ""},
#         "date":{"$gte":cond_date,"$type":"date"}
        "date":{"$type":"date"}
    }       
		return key
	
	def getReviewKey(self):
		# 增設條件if asin in [as1,as2,as3]
		# 增設條件 找出前三名
		# 預設查看前5個產品
		# Query review
		key = {
        "main_cat": {'$eq': self.main_cat},
        'big_categories': self.category1,
#         'small_categories': category2,
#         '$or': [
#                 { "small_categories":"Cell Phones" },
#                 {"small_categories":"Phones & tablets, Mobile phones"}
#         ],
        "vote": {"$gte": 1},
        'reviewTime':{"$gte":self.cond_date,"$type":"date"}
    }
		return key
	
	
# Keyboards
class Keyboards():
	def __init__(self):
		self.main_cat = 'All Electronics'
		self.category1 = 'Electronics'
		self.category2 ='Keyboards'   
# 		self.db_col='new_Product2'
		# self.cond_date = '2015-01-01'
		self.cond_date = datetime(1900, 1, 1, 0, 0, 0)
		
	def getAttr(self):
		return self.main_cat,self.category1,self.category2,self.cond_date
		
	def getProductKey(self):
		key = {
        "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
#         "main_cat": {'$eq': main_cat, "$type": "string", "$exists": True},
        "category1": {'$eq': self.category1, "$type": "string", "$exists": True},
#         "category2": {'$eq': category2, "$type": "string", "$exists": True},
        '$or': [
                {'category2':"Electronic Keyboards"},
                {'category2':"Portable Keyboards"},
                {'category2':"Keyboards"}  ],
        "title": {"$ne": "nan", "$exists": True, "$type": "string"},
        'salesRank': {"$exists": True,"$ne": ""},
#         "date":{"$gte":cond_date,"$type":"date"}
        "date":{"$type":"date"}
    }       
		return key
	
	def getReviewKey(self):
		# 增設條件if asin in [as1,as2,as3]
		# 增設條件 找出前三名
		# 預設查看前5個產品
		# Query review
		key = {
#         "main_cat": {'$eq': main_cat},
        'big_categories': self.category1,
#         'small_categories': category2,
        '$or': [
                {'small_categories':"Electronic Keyboards"},
                {'small_categories':"Portable Keyboards"},
                {'small_categories':"Keyboards"}  ],
        "vote": {"$gte": 1},
        'reviewTime':{"$gte":self.cond_date,"$type":"date"}
    }
		return key