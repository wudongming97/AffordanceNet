import os
import json
import pickle
import requests
from concurrent.futures import ThreadPoolExecutor

# Dataset configuration
DATASET = 'handal'

# Object categories with handle
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
    """
    Load a pickle file and extract data entries containing objects with handles,
    skipping already processed samples.
    """
    with open(pkl_path, 'rb') as f:
        val_data = pickle.load(f)

    filtered_data = []
    for class_name, img_list in val_data['images'].items():
        if class_name not in OBJECTS_WITH_HANDLE:
            continue
        for i, img_path in enumerate(img_list):
            class_label = val_data['class_names'][class_name][i]
            save_path = os.path.join(
                f'./reason_affordance/{DATASET}_hard_reasoning',
                class_label,
                os.path.splitext(os.path.basename(img_path))[0] + ".json"
            )
            if not os.path.exists(save_path):
                filtered_data.append({'img_name': img_path, 'class_name': class_label})

    return filtered_data


def process_sentence(category):
    """
    Generate reasoning instructions (<1>, <2>) from category name using GPT.
    """
    payload = {
        'model': 'gpt-4',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'system',
             'content': (
                 'Based on several words where the first is category name, please design an instruction <1> and instruction <2> in embodied scenes. '
                 'The instruction <1> must not include object category name itself. '
                 'The instruction <2> must include the object category name itself. '
                 'The instruction <2> must belong to embodied manipulation and give action if instruction <1> provides. '
                 'The instruction <2> does not exceed 50 words.'
             )},
            {'role': 'user', 'content': 'microwave, open'},
            {'role': 'assistant', 'content': '<1> Heat up food quickly. <2> The microwave is closed, so it can be open to access the food inside.'},
            {'role': 'user', 'content': 'knife'},
            {'role': 'assistant', 'content': '<1> I want to cut a bread. <2> The knife has a handle, you can use its handle to cut bread.'},
            {'role': 'user', 'content': 'computer mouse'},
            {'role': 'assistant', 'content': '<1> Give me a tool to control the cursor on the screen. <2> The computer mouse is here. It has no handle, so you can grasp its whole body.'},
            {'role': 'user', 'content': 'fork'},
            {'role': 'assistant', 'content': '<1> Use to pierce and lift food. <2> The fork is here, and its handle can be grasped.'},
            {'role': 'user', 'content': 'screwdrivers'},
            {'role': 'assistant', 'content': '<1> I need a tool to tighten or loosen screws. <2> The screwdriver is here, hold its handle to turn and control screws.'},
            {'role': 'user', 'content': category}
        ]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"[API Error] {category}: {response.status_code} - {response.text}")
        return None


def process_json(entry):
    """
    Process a single image/class entry by generating reasoning and saving result to file.
    """
    class_name = entry['class_name']

    for _ in range(5):
        result = process_sentence(class_name)
        if result and '<1>' in result and '<2>' in result:
            break
    else:
        print(f"[Retry Failed] {class_name}")
        return

    try:
        question = result.split('<2>')[0].split('<1>')[-1].strip()
        answer = result.split('<2>')[-1].strip()

        save_dir = os.path.join(f'./reason_affordance/{DATASET}_hard_reasoning', class_name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(entry['img_name']))[0] + ".json")
        output = {
            'img_name': entry['img_name'],
            'class_name': class_name,
            'question': question,
            'answer': answer
        }

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"[Saved] {save_path}")
    except Exception as e:
        print(f"[Error] Failed to save {class_name}: {e}")


def main():
    """
    Main execution: loads data, then processes in parallel.
    """
    pkl_path = f'./data/{DATASET}_val.pkl'
    entries = read_pkl_file(pkl_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(process_json, entries)


if __name__ == "__main__":
    main()
