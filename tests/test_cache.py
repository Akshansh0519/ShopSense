import json

from app.services.cache import RedisCache


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []

    def delete(self, key):
        self.commands.append(("delete", key))
        return self

    def zadd(self, key, mapping):
        self.commands.append(("zadd", key, mapping))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    def execute(self):
        for command in self.commands:
            if command[0] == "delete":
                self.client.store[command[1]] = {}
            elif command[0] == "zadd":
                self.client.store[command[1]] = command[2]


class FakeRedis:
    def __init__(self):
        self.store = {}

    def pipeline(self):
        return FakePipeline(self)

    def zrange(self, key, start, end, desc=False):
        items = list(self.store.get(key, {}).items())
        items.sort(key=lambda item: item[1], reverse=desc)
        values = [item[0] for item in items]
        return values[start:] if end == -1 else values[start : end + 1]


def test_cache_preserves_score_order():
    cache = RedisCache.__new__(RedisCache)
    cache.client = FakeRedis()
    cache.ttl = 60

    recs = [
        {"item_id": "a", "score": 0.2, "signals": {"popularity": 1.0}},
        {"item_id": "b", "score": 0.9, "signals": {"popularity": 1.0}},
    ]
    cache.set_recommendations("u1", "v1", recs)
    cached = cache.get_recommendations("u1", "v1")

    assert [item["item_id"] for item in cached] == ["b", "a"]
