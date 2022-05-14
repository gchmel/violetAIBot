import os
import json


class SmartHistory:

    def __init__(self, size, folder):
        if size is None:
            self.max_size = 10
        else:
            self.max_size = size

        self.size = 0
        self.folder = folder
        remove_history = False
        if os.path.isfile("./sources/" + folder + "/history.json"):
            with open(f'./sources/{self.folder}/history.json', 'r') as f:
                file = json.load(f)
                if (self.max_size == file['max_size']):
                    self.size = file['size']
                    self.history = file['history']
                else:
                    remove_history = True
        else:
            self.generate_new_history()

        if remove_history:
            os.remove(f'./sources/{self.folder}/history.json')
            self.generate_new_history()

    def generate_new_history(self):
        self.history = [0] * self.max_size
        data = '{ "max_size": 0, "size": 0, "history":[]}'
        file = json.loads(data)
        file['max_size'] = self.max_size
        file['size'] = self.size
        file['history'] = self.history
        json.dump(file, open(f'./sources/{self.folder}/history.json', 'w'), indent=2)

    def put(self, item):
        if self.size == self.max_size:
            self.history.remove(self.history[self.max_size - 1])
            temp = [item]
            temp.extend(self.history)
            self.history = temp
        else:
            self.history.remove(self.history[self.max_size - 1])
            temp = [item]
            temp.extend(self.history)
            self.history = temp
            self.size = self.size + 1
        with open(f'./sources/{self.folder}/history.json', 'r+') as f:
            file = json.load(f)
            file['size'] = self.size
            file['history'] = self.history
            f.seek(0)
            json.dump(file, f, indent=2)
            f.truncate()

    def get(self):
        return self.history[0]

    def get(self, index):
        return self.history[index]

    def get_all(self):
        return self.history
