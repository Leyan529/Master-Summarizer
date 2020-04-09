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

'''
All Electronics + 713433
Pet Supplies + 677196
Sports & Outdoors + 1376309
Health & Personal Care + 115155
Camera & Photo + 450880
Cell Phones & Accessories 689380

# 未加入
Software + 116519
Arts, Crafts & Sewing + 267365
Musical Instruments + 236420
Tools & Home Improvement 984019
Computers 653450
Software  116519
Automotive  680817
Office Products  359332
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