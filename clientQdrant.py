
from dbqdrant import DbQdrant
import json

class ClientDbQdrant:
    _db : DbQdrant
    def __init__(self, db) -> None:
        _db = db

    def chat(self, message, system):
        res = self._db.search(message,2)
        print(res[0])
        print(res[0].id)
        print(res[0].score)
        print(res[0].payload)
        team1 = res[0].payload['label']
        team2 = res[1].payload['label']
        return json.dump({ "team1" : team1, "team2" : team2})