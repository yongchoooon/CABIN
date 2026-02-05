import argparse
import os
import json
from tqdm import tqdm
from openai import OpenAI
from prompt_generation.data_info import data_info
from prompt_generation.prompt_for_llm import prompt
from dotenv import load_dotenv
load_dotenv()

def get_response(client, modelname, prompt, classes, domain):
    total_results = {}
    for i, classname in enumerate(tqdm(classes)):
        class_prompt = prompt.format(classname=classname, domain=domain)
        completion = client.chat.completions.create(
          model=modelname,
          messages=[
            {"role": "user", "content": class_prompt}
          ],
          response_format={"type": "json_object"},
          temperature=0.0,
          seed=42,
        )
        output = json.loads(completion.choices[0].message.content)
        total_results[classname] = output
        if (i + 1) % 10 == 0:
            print(f"CLASS: {classname} || output : {output}")
    return total_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='food', choices=['food', 'cub', 'eurosat', 'flowers', 'caltech', 'pets'])
    parser.add_argument('--savedir', type=str, default='prompt_generation/prompt_templates')
    parser.add_argument('--modelname', type=str, default='gpt-4o-2024-11-20')
    args = parser.parse_args()
    
    openai_token = os.environ.get("CABIN_OPENAI_TOKEN")

    os.makedirs(args.savedir, exist_ok=True)
    
    client = OpenAI(api_key=openai_token)
    modelname = args.modelname
    dataname = args.dataname
    domain = data_info[dataname]['domain']
    classes = data_info[dataname]['classes']
    
    total_results = get_response(client, modelname, prompt, classes, domain)
    save_path = os.path.join(args.savedir, dataname + ".json")
    
    with open(save_path, 'w', encoding='UTF-8') as file:
        file.write(json.dumps(total_results, ensure_ascii=False))
    
    
    