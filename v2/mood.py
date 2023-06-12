import json
import os


class Mood:

    def __init__(self, folder: str, username: str):

        # default values
        self.folder = folder
        self.username = username
        self.x_max = 100
        self.y_max = 100
        self.x_threshold = 0.3
        self.y_threshold = 0
        self.x = 0
        self.y = 0

        try:
            os.makedirs("./sources/mood/")
        except FileExistsError:
            pass

        if os.path.isfile("./sources/" + self.folder + "/mood/" + self.username + "_mood.json"):
            with open(f'./sources/{self.folder}/mood/{self.username}_mood.json', 'r') as f:
                file = json.load(f)
                self.x_max = file["x_max"]
                self.y_max = file["y_max"]
                self.x_threshold = file["x_threshold"]
                self.y_threshold = file["y_threshold"]
                self.x = file["x"]
                self.y = file["y"]
        else:
            data = {
                "username": self.username,
                "x_max": self.x_max,
                "y_max": self.y_max,
                "x_threshold": self.x_threshold,
                "y_threshold": self.y_threshold,
                "x": self.x,
                "y": self.y
            }
            json.dump(data, open(f'./sources/{self.folder}/mood/{self.username}_mood.json', 'w'), indent=2)

    def save(self) -> None:
        with open(f'./sources/{self.folder}/mood/{self.username}_mood.json', 'r') as f:
            file = json.load(f)
            file["x"] = self.x
            file["y"] = self.y
            json.dump(file, open(f'./sources/{self.folder}/mood/{self.username}_mood.json', 'w'), indent=2)

    def update_mood(self, x_change: float, y_change: float):
        self.x = self.x + x_change
        self.y = self.y + y_change
        self.save()

    def get_mood(self) -> str:
        if abs(self.x) < self.x_max * self.x_threshold:
            return "neutral"
        elif self.x_max * self.x_threshold < self.x < 2 * self.x_max * self.x_threshold:
            return "happy"
        elif -1 * self.x_max * self.x_threshold > self.x > -2 * self.x_max * self.x_threshold:
            return "sad"
        elif self.x > 2 * self.x_max * self.x_threshold:
            if self.y > self.y_max * self.y_threshold:
                return "angry"
            elif self.y < self.y_max * self.y_threshold:
                return "depressed"
        elif self.x < -2 * self.x_max * self.x_threshold:
            if self.y > self.y_max * self.y_threshold:
                return "love"
            elif self.y < self.y_max * self.y_threshold:
                return "best_friends"
