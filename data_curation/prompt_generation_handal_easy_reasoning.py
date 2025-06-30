import os
import json
import pickle
import requests
from concurrent.futures import ThreadPoolExecutor

# Dataset name
DATASET = 'handal'

# Handle-equipped objects to filter
OBJECTS_WITH_HANDLE = [
    'strainers', 'fixed joint pliers', 'hammers', 'ladles', 'whisks', 'measuring cups',
    'locking pliers', 'power drills', 'adjustable wrenches', 'mugs', 'ratchets', 'utensils',
    'combinational wrenches', 'pots pans', 'spatulas', 'screwdrivers', 'slip joint pliers'
]

# OpenAI API settings (update key!)
API_URL = 'https://api.openai.com/v1/chat/completions'
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR-API-KEY'  # Replace with your real key
}


def read_pkl_file(pkl_path):
    """Reads pkl file and filters entries for objects with handles."""
    with open(pkl_path, 'rb') as f:
        val_data = pickle.load(f)

    filtered_data = []
    for class_name, image_list in val_data['images'].items():
        if class_name in OBJECTS_WITH_HANDLE:
            for idx, img in enumerate(image_list):
                class_label = val_data['class_names'][class_name][idx]
                save_path = os.path.join(
                    f'./reason_affordance/{DATASET}_easy_reasoning',
                    class_label,
                    os.path.splitext(os.path.basename(img))[0] + ".json"
                )
                if not os.path.exists(save_path):
                    filtered_data.append({'img_name': img, 'class_name': class_label})
    return filtered_data


def process_sentence(class_name):
    """Send prompt to OpenAI and return generated sentence."""
    prompt = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'system',
         'content': (
             'Based on several words where the first is category name, '
             'please design an instruction <1> and instruction <2> in embodied scenes. '
             'The instruction <1> must include object category name itself. '
             'The instruction <2> must include the object category name itself. '
             'The instruction <2> must belong to embodied manipulation and give action if instruction <1> provides. '
             'The instruction <2> does not exceed 50 words.'
         )},
        {'role': 'user', 'content': 'mug'},
        {'role': 'assistant',
         'content': '<1> I need a drink. Please find a mug to fill water. <2> The mug has a handle as affordance map. So the robot can hold its handle.'},
        {'role': 'user', 'content': 'knife'},
        {'role': 'assistant',
         'content': '<1> Please give me a knife to cut apple. <2> The knife has a handle, and you can use its handle to cut apple.'},
        {'role': 'user', 'content': 'hammers'},
        {'role': 'assistant',
         'content': '<1> What is the proper way to hold the hammers? <2> The correct method is to hold the hammer by its handle.'},
        {'role': 'user', 'content': 'fork'},
        {'role': 'assistant',
         'content': '<1> Kindly pick up the fork. <2> You will be holding the fork handle.'},
        {'role': 'user', 'content': 'screwdrivers'},
        {'role': 'assistant',
         'content': '<1> I need a tool to tighten or loosen screws. <2> The screwdriver is here, hold its handle to turn and control screws.'},
        {'role': 'user', 'content': class_name}
    ]

    response = requests.post(API_URL, headers=HEADERS, json={'model': 'gpt-4', 'messages': prompt})
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"API Error for {class_name}:", response.text)
        return None


def process_json(data):
    """Process a single data entry and save result to JSON file."""
    class_name = data["class_name"]

    # Retry up to 5 times
    for _ in range(5):
        result = process_sentence(class_name)
        if not result or '<1>' not in result or '<2>' not in result:
            continue
        break
    else:
        print(f"Failed to process: {class_name}")
        return

    print("Processed:", result)

    try:
        question = result.split('<2>')[0].split('<1>')[-1].strip()
        answer = result.split('<2>')[-1].strip()

        save_dir = os.path.join(f'./reason_affordance/{DATASET}_easy_reasoning', class_name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(data["img_name"]))[0] + ".json")
        output = {'img_name': data["img_name"], 'class_name': class_name, 'question': question, 'answer': answer}

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=4)

    except Exception as e:
        print(f"Error saving file for {class_name}:", e)


def main():
    pkl_file = f'./data/{DATASET}_val.pkl'
    data_list = read_pkl_file(pkl_file)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(process_json, data_list)


if __name__ == "__main__":
    main()
