
from dbqdrant import DbQdrant
import json

class ClientDbQdrant:
    _db : DbQdrant
    def __init__(self, db) -> None:
        self._db = db

    def chat(self, message):
        res = self._db.search_topic(message,self._db.team_name,2)
        print(res[0])
        print(res[0].id)
        print(res[0].score)
        print(res[0].payload)
        team1 = res[0].payload['label']
        team2 = res[1].payload['label']
        return json.dumps({ "team1" : team1, "team2" : team2})