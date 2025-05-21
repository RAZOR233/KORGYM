# eval/eval.py
import asyncio
import os
import logging
import re
import random
import requests
import json

import pandas as pd
from tqdm import tqdm
import tiktoken

from .utils import parse_init
from .eval_lib import predict, save_process

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

game_dict = {
    '1-DateCount':'single','2-GuessWord':'single','3-2048':'multiple','4-SudoKu':'single',
    '5-light_out_game':'single','8-word_puzzle':'single','9-Jigsaw_puzzle':'single',
    '10-minigrid':'multiple','11-maze':'single','12-sokoban':'single','13-play_lines':'single',
    '15-emoji_connect':'single','16-jiafa':'single','17-fill_game':'single','18-alien':'single',
    '19-party_time':'single','20-city_path':'single','21-Anagramania':'single',
    '22-alphabetical_sorting':'single','23-puzzlegame':'single','24-snake':'multiple',
    '25-Tetris':'multiple','26-TrustRovolution':'multiple','27-NpointPlus':'multiple',
    '28-word_encryption':'single','29-Construction_Company':'single','30-Tower_of_Hanoi':'multiple',
    '31-ball_arrange':'multiple','32-numeral_bricks':'single','33-wordle':'multiple',
    '34-one_touch_drawing':'single','35-pipe_game':'single','36-CryptoWord':'multiple',
    '37-SpiderSolitaire':'multiple','38-minesweeper':'multiple','39-Nullify':'multiple',
    '40-CircleTheCat-Text':'multiple','41-PVZ':'multiple','42-diagram_coloring':'single',
    '43-CircleTheCat-Multimodal':'multiple','44-city':'single','47-free_the_key':'multiple',
    '48-map_position_simulation_text':'single','49_map_position_simulation_multimodal':'single',
    '50-SudoKu_MultiModal':'single','51-ball_arrange_multimodal':'multiple',
    '52-wordle_multimodal':'multiple','53-Arrow-pathway':'single','54-jiafa_multimodal':'single','55-LongCat':'single',
    '56-black_white_copy':'single'
}

def normalize_response(response: str) -> str:
    return (
        response.replace("**", "")
                .replace("$\\boxed{", "")
                .replace("}$", "")
                .replace("\\$", "")
                .replace("$\\text{", "")
                .replace("$", "")
                .replace("\\mathrm{", "")
                .replace("\\{", "")
                .replace("\\text", "")
                .replace("\\(", "")
                .replace("\\mathbf{", "")
                .replace("{", "")
                .replace("\\boxed", "")
    )

def get_prompt0_response(ori_answer):
    if ori_answer is None:
        return ""
    gen = normalize_response(ori_answer)
    pos = gen.lower().rfind("answer")
    if pos == -1:
        return ""
    gen = gen[pos:]
    pattern = r"(?i)Answer\s*:\s*(.*)"
    match = re.findall(pattern, gen)
    return match[-1] if match else ""

def generate(url, seed, level=4):
    return requests.post(f"{url}/generate", json={"seed": seed}).json()

def print_board(url, item):
    return requests.post(f"{url}/print_board", json=item).json()['board']

def verify(url, item):
    try:
        resp = requests.post(f"{url}/verify", json=item, timeout=30)
        resp.raise_for_status()
        item.update(resp.json())
    except Exception:
        item['score'] = 0
    return item

async def eval_single_file(output_dir, model_name, address, key, sem, game_name, level, url):
    checkpoint_dir = os.path.join(output_dir, game_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(checkpoint_dir, f"{model_name}_{game_name}_level{level}_checkpoint.jsonl")

    processed = []
    seen = set()
    if os.path.exists(ckpt):
        with open(ckpt, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                processed.append(d)
                seen.add(d['seed'])

    to_run = []
    for seed in range(50):
        if seed in seen:
            continue
        item = generate(url, seed, level)
        item['seed'] = seed
        item['response'] = []
        item['prompt'] = print_board(url, item)
        to_run.append(item)

    if to_run:
        results = await predict(to_run, sem, model_name, address, key)
        for item in results:
            item['action'] = get_prompt0_response(item['response'][-1])
            item = verify(url, item)
            processed.append(item)
            with open(ckpt, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    file_name = f"{model_name}_{game_name}_level{level}"
    final_dir = os.path.join(output_dir, game_name)
    save_process(processed, final_dir, file_name)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    logging.info(f"Complete the evaluation of the file: {file_name}")

async def eval_file(output_dir, model_name, address, key, sem, game_name, level, url):
    checkpoint_dir = os.path.join(output_dir, game_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(checkpoint_dir, f"{model_name}_{game_name}_level{level}_checkpoint.json")

    if os.path.exists(ckpt):
        with open(ckpt, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        count = state['count']+1
        item_list = state['item_list']
        final_list = state['final_list']
        print(f"loading checkpoint:{count}")
    else:
        count = 1
        final_list = []
        item_list = []
        for seed in range(20):
            item = generate(url, seed, level)
            item['seed'] = seed
            item['response'] = []
            item['prompt'] = print_board(url, item)
            item_list.append(item)

    while count <= 100:
        tqdm.write(f'round {count}')
        item_list = await predict(item_list, sem, model_name, address, key)
        i = len(item_list) - 1
        while i >= 0:
            itm = item_list[i]
            itm['action'] = get_prompt0_response(itm['response'][-1])
            itm = verify(url, itm)
            itm['prompt'] = print_board(url, itm)
            if itm.get('is_end'):
                final_list.append(item_list.pop(i))
            i -= 1

        with open(ckpt, 'w', encoding='utf-8') as f:
            json.dump({'count': count, 'item_list': item_list, 'final_list': final_list}, f, ensure_ascii=False)

        if not item_list:
            break
        count += 1

    final_list.extend(item_list)
    file_name = f"{model_name}_{game_name}_level{level}"
    final_dir = os.path.join(output_dir, game_name)
    save_process(final_list, final_dir, file_name)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    logging.info(f"Complete the evaluation of the file: {file_name}")

async def main():
    sem = asyncio.Semaphore(10)
    args = parse_init()
    if game_dict.get(args.game) == 'single':
        await eval_single_file(args.output, args.model, args.address, args.key, sem, args.game, args.level, args.url)
    else:
        await eval_file(args.output, args.model, args.address, args.key, sem, args.game, args.level, args.url)

if __name__ == "__main__":
    asyncio.run(main())
