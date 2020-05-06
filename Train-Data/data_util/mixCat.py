import dateutil.parser as parser
from datetime import datetime


# All Electronics + Pet Supplies + Sports & Outdoors + Health & Personal Care + Camera & Photo + Cell Phones & Accessories
# 未加入 # Software + Arts, Crafts & Sewing + Musical Instruments + Tools & Home Improvement
class Mix6():     
    def __init__(self):
        self.main_cat = 'Mix6_mainCat' # 4022353

    def getAttr(self):
        return self.main_cat

    def getProductKey(self):
        key = {
            "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
            # "main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
            "date": {"$type": "date"},
            '$or': [
                { "main_cat":"All Electronics"},
                { "main_cat":"Pet Supplies"},
                { "main_cat":"Sports & Outdoors"},
                { "main_cat":"Health & Personal Care"},
                { "main_cat":"Camera & Photo"},
                { "main_cat":"Cell Phones & Accessories"},
            ]
            # "feature": {"$type": "array"}
        }
        return key

    def getReviewKey(self):
        # 增設條件if asin in [as1,as2,as3]
        # 增設條件 找出前三名
        # 預設查看前5個產品
        # Query review
        key = {
            # "main_cat": {'$eq': self.main_cat},
            #             'big_categories': self.category1,
            #         'small_categories': category2,
            '$or': [
                { "main_cat":"All Electronics"},
                { "main_cat":"Pet Supplies"},
                { "main_cat":"Sports & Outdoors"},
                { "main_cat":"Health & Personal Care"},
                { "main_cat":"Camera & Photo"},
                { "main_cat":"Cell Phones & Accessories"}
            ],
            "vote": {"$gte": 1},
            #             'reviewTime':{"$gte":self.cond_date,"$type":"date"}
        }
        return key

class Mix12():     
    def __init__(self):
        self.main_cat = 'Mix12_mainCat' # 4022353

    def getAttr(self):
        return self.main_cat

    def getProductKey(self):
        key = {
            "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
            # "main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
            "date": {"$type": "date"},
            '$or': [
                { "main_cat":"All Electronics"},
                { "main_cat":"Pet Supplies"},
                { "main_cat":"Sports & Outdoors"},
                { "main_cat":"Health & Personal Care"},
                { "main_cat":"Camera & Photo"},
                { "main_cat":"Cell Phones & Accessories"},

                { "main_cat":"Computers"},
                { "main_cat":"Automotive"},
                { "main_cat":"Musical Instruments"},
                { "main_cat":"Tools & Home Improvement"},
                { "main_cat":"Arts, Crafts & Sewing"},
                { "main_cat":"Office Products"}
            ]
            # "feature": {"$type": "array"}
        }
        return key

    def getReviewKey(self):
        # 增設條件if asin in [as1,as2,as3]
        # 增設條件 找出前三名
        # 預設查看前5個產品
        # Query review
        key = {
            # "main_cat": {'$eq': self.main_cat},
            #             'big_categories': self.category1,
            #         'small_categories': category2,
            '$or': [
                { "main_cat":"All Electronics"},
                { "main_cat":"Pet Supplies"},
                { "main_cat":"Sports & Outdoors"},
                { "main_cat":"Health & Personal Care"},
                { "main_cat":"Camera & Photo"},
                { "main_cat":"Cell Phones & Accessories"},

                { "main_cat":"Computers"},
                { "main_cat":"Automotive"},
                { "main_cat":"Musical Instruments"},
                { "main_cat":"Tools & Home Improvement"},
                { "main_cat":"Arts, Crafts & Sewing"},
                { "main_cat":"Office Products"}
            ],
            "vote": {"$gte": 1},
            #             'reviewTime':{"$gte":self.cond_date,"$type":"date"}
        }
        return key

class OtherTest():     
    def __init__(self):
        self.main_cat = 'OtherTest' # 4022353

    def getAttr(self):
        return self.main_cat

    def getProductKey(self):
        key = {
            "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
            # "main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
            "date": {"$type": "date"},
            '$or': [
                { "main_cat":"Software"},
                { "main_cat":"Video Games"}              
            ]
            # "feature": {"$type": "array"}
        }
        return key

    def getReviewKey(self):
        # 增設條件if asin in [as1,as2,as3]
        # 增設條件 找出前三名
        # 預設查看前5個產品
        # Query review
        key = {
            # "main_cat": {'$eq': self.main_cat},
            #             'big_categories': self.category1,
            #         'small_categories': category2,
            '$or': [
                { "main_cat":"Software"},
                { "main_cat":"Video Games"} 
            ],
            "vote": {"$gte": 1},
            #             'reviewTime':{"$gte":self.cond_date,"$type":"date"}
        }
        return key


'''
Mix10
All Electronics + 713433
Pet Supplies + 677196
Sports & Outdoors + 1376309
Health & Personal Care + 115155
Camera & Photo + 450880
Cell Phones & Accessories 689380

# 加入 Mix10
Computers 653450
Automotive  680817
Musical Instruments + 236420
Tools & Home Improvement 984019
Arts, Crafts & Sewing + 267365
Office Products  359332

# 未加入
Software  116519

Video Games 129331
Books  9298896
'''

'''
All Electronics + Pet Supplies + Sports & Outdoors + Health & Personal Care + Camera & Photo + Cell Phones & Accessories
db.getCollection('new_reviews2').find(
    {
        $or: [ 
            { main_cat: "All Electronics" }, 
            { main_cat: "Pet Supplies" }, 
            { main_cat: "Sports & Outdoors" }, 
            { main_cat: "Health & Personal Care" }, 
            { main_cat: "Camera & Photo" }, 
            { main_cat: "Cell Phones & Accessories" }            
            ],

        "vote": {"$gte": 1}}
     ).count()
'''




'''
main_cat = Mix6().getAttr()

'''

'''
db.getCollection('new_reviews2').distinct('main_cat',
{"reviewTime":{
    "$type":"date"},
    "salesRank":{"$ne":"nan","$exists": true},
    })


[
    "Arts, Crafts & Sewing",
    "Musical Instruments",
    "Cell Phones & Accessories",
    "Software",
    "Toys & Games",
    "Home Audio & Theater",
    "All Electronics",
    "Camera & Photo",
    "Amazon Home",
    "Computers",
    "Tools & Home Improvement",
    "Car Electronics",
    "Video Games",
    "Industrial & Scientific",
    "Pet Supplies",
    "Baby",
    "Office Products",
    "Sports & Outdoors",
    "Automotive",
    "",
    "Health & Personal Care",
    "All Beauty",
    "Movies & TV",
    "GPS & Navigation",
    "Books",
    "Portable Audio & Accessories",
    "Appliances",
    "Magazine Subscriptions",
    "Collectibles & Fine Art",
    "Collectible Coins",
    "Entertainment",
    "Fine Art",
    "Grocery",
    "Gift Cards",
    "Luxury Beauty",
    "MEMBERSHIPS & SUBSCRIPTIONS",
    "Amazon Devices",
    "Apple Products",
    "Beats by Dr. Dre",
    "Fire Phone",
    "3D Printing",
    "Amazon Fire TV"
]
'''

'''
db.getCollection('new_reviews2').distinct('big_categories',
{"reviewTime":{
    "$type":"date"},
    "salesRank":{"$ne":"nan","$exists": true},
    })


[
    "Musical Instruments",
    "Magazine Subscriptions",
    "Collectibles & Fine Art",
    "Industrial & Scientific",
    "Appliances",
    "Gift Cards",
    "Arts, Crafts & Sewing",
    "Digital Music",
    "Handmade",
    "Video Games",
    "Software",
    *"Home & Kitchen",
    *"Electronics",
    *"Clothing, Shoes & Jewelry",
    "Sports & Outdoors",
    "Grocery & Gourmet Food",
    "Movies & TV",
    *"Cell Phones & Accessories",
    *"Automotive",
    "CDs & Vinyl",
    "Office Products",
    "Patio, Lawn & Garden",
    "Pet Supplies",
    "Tools & Home Improvement",
    "Toys & Games",
    "Books"
]
'''
class Mixbig_5():     
    def __init__(self):
        self.main_cat = 'Mixbig_5' # 8502433

    def getAttr(self):
        return self.main_cat

    def getProductKey(self):
        key = {
            "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
            # "main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
            "date": {"$type": "date"},
            '$or': [
                { "big_categories":"Home & Kitchen"},
                { "big_categories":"Electronics"}, 
                { "big_categories":"Clothing, Shoes & Jewelry"}, 
                { "big_categories":"Cell Phones & Accessories"}, 
                { "big_categories":"Automotive"}
            ]
            # "feature": {"$type": "array"}
        }
        return key

    def getReviewKey(self):
        # 增設條件if asin in [as1,as2,as3]
        # 增設條件 找出前三名
        # 預設查看前5個產品
        # Query review
        key = {
            # "main_cat": {'$eq': self.main_cat},
            #             'big_categories': self.category1,
            #         'small_categories': category2,
            '$or': [
                { "big_categories":"Home & Kitchen"},
                { "big_categories":"Electronics"}, 
                { "big_categories":"Clothing, Shoes & Jewelry"}, 
                { "big_categories":"Cell Phones & Accessories"}, 
                { "big_categories":"Automotive"}
            ],
            "vote": {"$gte": 1},
            #             'reviewTime':{"$gte":self.cond_date,"$type":"date"}
        }
        return key


# db.getCollection('new_reviews2').find({'big_categories':'Electronics'}).count()
# 2421693

# db.getCollection('new_reviews2').distinct('main_cat',
# {"reviewTime":{
#     "$type":"date"},
#     "salesRank":{"$ne":"nan","$exists": true},
#     'big_categories':'Electronics'
#     })

# [
#     "Books",
#     "All Electronics",
#     "Computers",
#     "Cell Phones & Accessories",
#     "Home Audio & Theater",
#     "Portable Audio & Accessories",
#     "Baby",
#     "Tools & Home Improvement",
#     "Office Products",
#     "Arts, Crafts & Sewing",
#     "Amazon Home",
#     "Musical Instruments",
#     "Camera & Photo",
#     "GPS & Navigation",
#     "Car Electronics",
#     "Sports & Outdoors",
#     "Automotive",
#     "Movies & TV",
#     "Industrial & Scientific",
#     "Health & Personal Care",
#     "",
#     "Toys & Games",
#     "Video Games",
#     "Software",
#     "Pet Supplies",
#     "Amazon Devices",
#     "All Beauty",
#     "Grocery",
#     "Appliances",
#     "Apple Products",
#     "Beats by Dr. Dre",
#     "Gift Cards"
# ]

class Mixbig_Elect_30():     
    def __init__(self):
        self.main_cat = 'Mixbig_Elect_30' # 2405872

    def getAttr(self):
        return self.main_cat

    def getProductKey(self):
        key = {
            "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
            "big_categories":"Electronics",
            # "main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
            "date": {"$type": "date"},
            '$or': [
                { "main_cat":"Computers"},
                { "main_cat":"Cell Phones & Accessories"}, 
                { "main_cat":"Home Audio & Theater"}, 
                { "main_cat":"Office Products"}, 
                { "main_cat":"Musical Instruments"}, 
                { "main_cat":"Camera & Photo"},
                { "main_cat":"GPS & Navigation"},
                { "main_cat":"Car Electronics"},
                { "main_cat":"Automotive"},
                { "main_cat":"Health & Personal Care"},
                { "main_cat":"Apple Products"},
                { "main_cat":"Toys & Games"},
                { "main_cat":"Video Games"},
                { "main_cat":"Movies & TV"},
                { "main_cat":"Sports & Outdoors"},

                { "main_cat":"All Electronics"},
                { "main_cat":"Portable Audio & Accessories"},
                { "main_cat":"Baby"},
                { "main_cat":"Tools & Home Improvement"},
                { "main_cat":"Arts, Crafts & Sewing"},
                { "main_cat":"Amazon Home"},

                { "main_cat":"Toys & Games"},
                { "main_cat":"Video Games"},
                { "main_cat":"Software"},
                { "main_cat":"Pet Supplies"},

                { "main_cat":"Amazon Devices"},
                { "main_cat":"All Beauty"},
                { "main_cat":"Grocery"},
                { "main_cat":"Appliances"},
                { "main_cat":"Gift Cards"}
            ]
            # "feature": {"$type": "array"}
        }
        return key

    def getReviewKey(self):
        # 增設條件if asin in [as1,as2,as3]
        # 增設條件 找出前三名
        # 預設查看前5個產品
        # Query review
        key = {
            "big_categories":"Electronics",
            # "main_cat": {'$eq': self.main_cat},
            #             'big_categories': self.category1,
            #         'small_categories': category2,
            '$or': [
                { "main_cat":"Computers"},
                { "main_cat":"Cell Phones & Accessories"}, 
                { "main_cat":"Home Audio & Theater"}, 
                { "main_cat":"Office Products"}, 
                { "main_cat":"Musical Instruments"}, 
                { "main_cat":"Camera & Photo"},
                { "main_cat":"GPS & Navigation"},
                { "main_cat":"Car Electronics"},
                { "main_cat":"Automotive"},
                { "main_cat":"Health & Personal Care"},
                { "main_cat":"Apple Products"},
                { "main_cat":"Toys & Games"},
                { "main_cat":"Video Games"},
                { "main_cat":"Movies & TV"},
                { "main_cat":"Sports & Outdoors"},

                { "main_cat":"All Electronics"},
                { "main_cat":"Portable Audio & Accessories"},
                { "main_cat":"Baby"},
                { "main_cat":"Tools & Home Improvement"},
                { "main_cat":"Arts, Crafts & Sewing"},
                { "main_cat":"Amazon Home"},

                { "main_cat":"Toys & Games"},
                { "main_cat":"Video Games"},
                { "main_cat":"Software"},
                { "main_cat":"Pet Supplies"},

                { "main_cat":"Amazon Devices"},
                { "main_cat":"All Beauty"},
                { "main_cat":"Grocery"},
                { "main_cat":"Appliances"},
                { "main_cat":"Gift Cards"}
            ],
            "vote": {"$gte": 1},
            #             'reviewTime':{"$gte":self.cond_date,"$type":"date"}
        }
        return key

class Pure_kitchen():     
    def __init__(self):
        self.main_cat = 'Pure_kitchen' # 2405872

    def getAttr(self):
        return self.main_cat

    def getProductKey(self):
        key = {
            "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
            "big_categories":"Home & Kitchen",
            # "main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
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
            "big_categories":"Home & Kitchen",
            # "main_cat": {'$eq': self.main_cat},
            #             'big_categories': self.category1,
            #         'small_categories': category2,            
            "vote": {"$gte": 1},
            #             'reviewTime':{"$gte":self.cond_date,"$type":"date"}
        }
        return key

class Mixbig_Books_3():     
    def __init__(self):
        self.main_cat = 'Mixbig_Books_3' # 2405872

    def getAttr(self):
        return self.main_cat

    def getProductKey(self):
        key = {
            "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
            # "big_categories":"Home & Kitchen",
            # "main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
            "date": {"$type": "date"},            
            # "feature": {"$type": "array"}
            '$or': [
                { "big_categories":"Magazine Subscriptions"},
                { "big_categories":"Books"},
                { "big_categories":"CDs & Vinyl"},
            ]
        }
        return key

    def getReviewKey(self):
        # 增設條件if asin in [as1,as2,as3]
        # 增設條件 找出前三名
        # 預設查看前5個產品
        # Query review
        key = {
            # "big_categories":"Home & Kitchen",
            # "main_cat": {'$eq': self.main_cat},
            #             'big_categories': self.category1,
            #         'small_categories': category2,            
            "vote": {"$gte": 5},
            #             'reviewTime':{"$gte":self.cond_date,"$type":"date"}
            '$or': [
                { "big_categories":"Magazine Subscriptions"},
                { "big_categories":"Books"},
                { "big_categories":"CDs & Vinyl"},
            ]
        }
        return key

class Pure_Cloth():     
    def __init__(self):
        self.main_cat = 'Pure_Cloth' # 2405872

    def getAttr(self):
        return self.main_cat

    def getProductKey(self):
        key = {
            "description": {'$ne': 'NaN', "$exists": True, "$type": "array"},
            "big_categories":"Clothing, Shoes & Jewelry",
            # "main_cat": {'$eq': self.main_cat, "$type": "string", "$exists": True},
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
            "big_categories":"Clothing, Shoes & Jewelry",
            # "main_cat": {'$eq': self.main_cat},
            #             'big_categories': self.category1,
            #         'small_categories': category2,            
            "vote": {"$gte": 1},
            #             'reviewTime':{"$gte":self.cond_date,"$type":"date"}
        }
        return key
