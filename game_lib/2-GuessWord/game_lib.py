import random
import string
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse

def parse_init():
    """
    定义并解析eval代码的命令行参数，配置日志记录，并检查输入的数据文件目录和输出的目录是否存在。
    """
    parser = argparse.ArgumentParser(description="Data creation utility")

    # 添加命令行参数
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    # 添加命令行参数
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    # 解析命令行参数
    args = parser.parse_args()
    return args
app = FastAPI()
game_prompt='''
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: 1992/05/18'

{question}
'''
def print_board(item):
    rules_desc = []
    for pos, letter in item['rules']:
        rules_desc.append(f"the letter at position {pos+1} is '{letter}'")
    
    rules_text = " and ".join(rules_desc)
    board=f"Please provide an English word that meets the following requirements:\n1. The word must be {item['length']} letters long\n2. {rules_text}\n"
    prompt = game_prompt.format(question=board)
    return prompt

def get_valid_words(length, rules, word_list):
    """
    获取所有符合长度和规则的有效单词
    """
    valid_words = set()
    for word in word_list:
        if len(word.lower()) == length:
            match = True
            for pos, letter in rules:
                if pos >= length or word[pos].lower() != letter.lower():
                    match = False
                    break
            if match:
                valid_words.add(word.lower())
    return valid_words

def generate(seed):
    """
    生成符合规则的问题
    """
    words = []
    with open("words.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            # 过滤掉长度为2的单词
            if len(line) <= 4:
                continue
            words.append(line)
    word_list = set(words)

    if seed is not None:
        random.seed(seed)

    # 外层循环不断尝试生成新的 length 和 rules_num 直到成功
    while True:
        length = random.randint(5, 10)
        rules_num = random.randint(3, 4)
        rules = []
        attempts = 0
        max_attempts = 100

        while len(rules) < rules_num and attempts < max_attempts:
            # 随机选择位置和字母
            pos = random.randint(0, length - 1)
            letter = random.choice(string.ascii_lowercase)

            # 确保位置不重复
            if not any(pos == p for p, _ in rules):
                temp_rules = rules + [(pos, letter)]
                # 检查是否存在符合所有规则的单词
                valid_words = get_valid_words(length, temp_rules, word_list)
                if valid_words:
                    rules = temp_rules
            attempts += 1

        # 如果成功生成了足够的规则，则退出外层循环
        if len(rules) == rules_num:
            break

    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    item['current_valid_words'] = get_valid_words(length, rules, word_list)
    item['length'] = length
    item['rules'] = rules
    return item


def verify(item):
    """
    验证答案是否正确
    """
    words = []
    with open("verify_words.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            # 过滤掉长度为2的单词
            if len(line) <= 4:
                continue
            words.append(line)
    word_list = set(words)
    answer=item['action']
    
    if len(answer) != item['length']:
        item['score'] = 0
        return item
        
    if answer.lower() not in word_list:
        item['score'] = 0
        return item
        
    # 检查是否符合所有规则
    for pos, letter in item['rules']:
        if answer[pos].lower() != letter.lower():
            item['score'] = 0
            return item
    item['score'] = 1
    return item

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    current_valid_words: list
    length: int
    rules: list
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int
# 生成初始游戏状态
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    state['rules'] = [tuple(rule) for rule in state['rules']]
    board_output = print_board(state)
    return {"board": board_output}


# 生成初始游戏状态
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    return game_state

# 根据动作更新游戏状态
@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    # 从请求中获取游戏状态，并设置新的动作
    state = request.dict()
    state['rules'] = [tuple(rule) for rule in state['rules']]
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)

# # 使用示例
# if __name__ == "__main__":
#     item = generate(2442)
#     item['action']='employment'
#     print(print_board(item))
#     item=verify(item)
#     print(f"score: {item['score']}")