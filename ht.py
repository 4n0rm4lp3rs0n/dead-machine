class HashTable:
    def __init__(self, size = 128):
        self.size = size
        self.count = 0
        self.table = [[] for _ in range(self.size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def _rehash(self):
        old_table = self.table
        self.size *= 2
        self.table = [[] for _ in range(self.size)]
        self.count = 0

        for bucket in old_table:
            for key, value in bucket:
                self.insert(key, value)
    
    def insert(self, key, value):
        if self.count / self.size >= 0.5:
            self._rehash()

        idx = self._hash(key)
        bucket = self.table[idx]

        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        bucket.append((key, value))
        self.count += 1

    def get(self, key):
        idx = self._hash(key)
        bucket = self.table[idx]

        for k, v in bucket:
            if k == key:
                return v
        return None

