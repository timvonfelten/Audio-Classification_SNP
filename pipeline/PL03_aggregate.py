from pymongo import MongoClient

def aggregate_audio_data(mongo_uri, db_name, source_collection_name, target_collection_name, confidence_threshold):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[source_collection_name]
    aggregated_collection = db[target_collection_name]

    pipeline = [
        {"$unwind": "$Results"},
        {"$project": {
            "Filename": 1,
            "RMS_dB": "$Results.RMS_dB",  # RMS_dB-Wert hinzufügen
            "ModelResults": "$Results.Models"
        }},
        {"$project": {
            "Filename": 1,
            "RMS_dB": 1,  # RMS_dB in der Projektion behalten
            "stille": {"$cond": [{"$and": [{"$eq": ["$ModelResults.stille.Predicted Class", 1]},
                                        {"$gte": ["$ModelResults.stille.Confidence", confidence_threshold]}]}, 1, 0]},
            "vogel": {"$cond": [{"$and": [{"$eq": ["$ModelResults.vogel.Predicted Class", 1]},
                                        {"$gte": ["$ModelResults.vogel.Confidence", confidence_threshold]}]}, 1, 0]},
            "laerm": {"$cond": [{"$and": [{"$eq": ["$ModelResults.laerm.Predicted Class", 1]},
                                        {"$gte": ["$ModelResults.laerm.Confidence", confidence_threshold]}]}, 1, 0]},
            "natur": {"$cond": [{"$and": [{"$eq": ["$ModelResults.natur.Predicted Class", 1]},
                                        {"$gte": ["$ModelResults.natur.Confidence", confidence_threshold]}]}, 1, 0]}
        }},
        {"$lookup": {  # Zusätzliche Informationen aus den Originaldokumenten holen
            "from": "audio_classification",
            "localField": "Filename",
            "foreignField": "Filename",
            "as": "additional_info"
        }},
        {"$unwind": "$additional_info"},  # Den Array aufschlüsseln
        {"$group": {
            "_id": "$Filename",
            "average_RMS_dB": {"$avg": "$RMS_dB"},  # Durchschnitt von RMS_dB berechnen
            "stille_count": {"$sum": "$stille"},
            "vogel_count": {"$sum": "$vogel"},
            "laerm_count": {"$sum": "$laerm"},
            "natur_count": {"$sum": "$natur"},
            "Date": {"$first": "$additional_info.Date"},
            "Time": {"$first": "$additional_info.Time"},
            "Weekday": {"$first": "$additional_info.Weekday"},
            "Place": {"$first": "$additional_info.Place"}
        }}
    ]

    aggregated_results = collection.aggregate(pipeline)

    # Speichern Sie die Ergebnisse in der neuen Kollektion
    for result in aggregated_results:
        aggregated_collection.insert_one(result)
