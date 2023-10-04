from tkinter import *
from PIL import ImageTk,Image
from tkinter import messagebox
import pymysql

def store(sr):
    collection = database["issued_book"]
    collection2 = database["defaulters"]
    y = collection.count_documents({"$expr":{"$eq":["$serial number",sr]}})
    if y==1:
        m = collection2.count_documents({"$expr":{"$eq":["$serial number",sr]}})
        x = collection.delete_one({"serial number":sr})
        collection = database["book_list"]
        x = collection.update_one({"serial number":sr},{"$set":{"available":"yes"}})
        if m==0:
            return True, sr
        else:
            x = collection2.find({"$expr":{"$eq":["$serial number",sr]}})
            fine =0
            for i in x:
                fine = i["fine"]
            x = collection2.delete_one({"serial number":sr})
            return fine, sr

    else:
        return False, " "

    
