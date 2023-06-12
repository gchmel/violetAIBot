import json
import math
import time

from violetBot import VioletBot


def load_data(filename):
    training_data = []

    with open(f'./sources/learning_materials/{filename}', 'r') as f:
        data = json.load(f)

    for key in data.keys():
        for prompt in data[key]["content"]:
            training_data.append(prompt["message"])

    print("[DEBUG]: Loaded Training Data")
    return training_data


def train(training_data):
    total_len = len(training_data)
    for i, item in enumerate(training_data):
        if i <= 1110:
            continue

        if i % 100 == 0:
            violet.train_model()
            violet.save_model()
            violet.load_model()
            print(f"[DEBUG]: Re-Trained model at {i} position out of {total_len}")

        start_time = time.perf_counter()
        train_bot(training_data[i - 1], training_data[i])
        print(f'[DEBUG]: Learning {i} out of {total_len} took ', math.ceil((time.perf_counter() - start_time) * 1000),
              "milliseconds to calculate")
        time.sleep(0.5)


def train_bot(message, answer):
    start_time = time.perf_counter()
    violet.train_with_prompts(message, answer)
    print('[DEBUG]: Learning took ', math.ceil((time.perf_counter() - start_time) * 1000),
          "milliseconds to calculate")


def write_training_data_to_intents(filename_data, filename_intents):
    with open(f'./sources/learning_materials/{filename_data}', 'r') as f:
        data = json.load(f)

    intents_json = json.loads(open(f'./sources/{filename_intents}').read())

    total_len = len(data.keys())
    for j, key in enumerate(data.keys()):
        content = data[key]["content"]
        for i, prompt in enumerate(content):
            if i != len(content) - 1:
                message = prompt['message']
                result = content[i+1]['message']
                new_entry = {
                    "tag": message,
                    "patterns": [
                        message
                    ],
                    "neutral_responses": [result],
                    "sad_responses": [result],
                    "happy_responses": [result],
                    "depressed_responses": [result],
                    "angry_responses": [result],
                    "love_responses": [result],
                    "best_friends_responses": [result],
                }

                intents_json.append(new_entry)
        print(f'[DEBUG]: Wrote {j} out of {total_len} conversations')

    json.dump(intents_json, open(f'./sources/{filename_intents}', 'r+'), indent=2)





if __name__ == '__main__':
    # with open('./log.txt', 'w', encoding='utf8') as logfile:
    # sys.stdout = logfile
    #data = load_data("train.json")
    write_training_data_to_intents("train.json", 'dev/intents.json')
    violet = VioletBot('./sources/dev/intents.json', model_name="dev", history_size=1)
    violet.train_model()
    violet.save_model()
    violet.load_model()
    #train(data)

