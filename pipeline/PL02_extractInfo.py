# date_processing.py

from pymongo import MongoClient
from datetime import datetime

def get_weekday(date):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days[date.weekday()]

def update_documents(mongo_uri, db_name, collection_name):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    for document in collection.find():
        filename = document['Filename']
        parts = filename.split('_')
        place = parts[0]
        date_str = parts[1]
        time_str = parts[2].split('.')[0]

        datetime_obj = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        date = datetime_obj.strftime('%Y-%m-%d')
        time = datetime_obj.strftime('%H:%M:%S')
        weekday = get_weekday(datetime_obj)

        collection.update_one({'_id': document['_id']}, {'$set': {'Place': place, 'Date': date, 'Time': time, 'Weekday': weekday}})

    client.close()

# So können Sie die Funktion in einem anderen Skript verwenden:
# update_documents("mongodb://localhost:27017/", "ClassifiedAudioSNP", "audio_classification")
