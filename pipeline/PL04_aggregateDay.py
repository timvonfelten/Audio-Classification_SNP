from pymongo import MongoClient

def aggregate_daily_audio_data(mongo_uri, db_name, source_collection_name, target_collection_name):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[source_collection_name]

    pipeline = [
        {
            "$group": {
                "_id": {"Date": "$Date", "Place": "$Place", "Weekday": "$Weekday"},
                "hours_info": {
                    "$push": {
                        "Time": "$Time",
                        "stille_count": "$stille_count",
                        "vogel_count": "$vogel_count",
                        "laerm_count": "$laerm_count",
                        "natur_count": "$natur_count",
                        "average_RMS_dB": "$average_RMS_dB"
                    }
                }
            }
        },
        {
            "$addFields": {
                "hours_info": {
                    "$sortArray": {
                        "input": "$hours_info",
                        "sortBy": {"Time": 1}
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "Date": "$_id.Date",
                "Weekday": "$_id.Weekday",
                "Place": "$_id.Place",
                "hours_info": "$hours_info"
            }
        },
        {
            "$out": "aggregated_files_per_day"  # Speichern Sie das Ergebnis in einer neuen Kollektion
        }
    ]

    # Führen Sie die Aggregationspipeline aus
    collection.aggregate(pipeline)

    client.close()
