import random
from datetime import datetime, timedelta
import json
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
# 辅助函数：打印游戏板（用于记录每一步的地图状态）
def print_board(item):
    prompt = game_prompt.format(question=item['current_problem'])
    return prompt
        
def generate(seed=None):
    """
    生成一个随机的日期问题
    参数:
        seed: 随机数种子
    返回:
        date, offset, correct_answer: 当前日期， 偏移的日期， 正确答案
    """
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch' : 1,
    }
    if seed is not None:
        random.seed(seed)
        
    # 随机生成一个基准日期（1900-2100年之间）
    year = random.randint(500, 1525)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  
    
    # 随机生成偏移天数（-10到10天之间）
    offset = random.randint(-100000, 100000)
    
    # 创建日期对象
    base_date = datetime(year, month, day)
    target_date = base_date + timedelta(days=offset)  # 目标日期
    
    # 获取正确答案
    # self.correct_answer = self.weekdays[target_date.weekday()]
    item['correct_answer'] = target_date.strftime("%Y/%m/%d")
    
    # 构造问题
    direction = "ago" if offset > 0 else "later"
    abs_offset = abs(offset)
    
    item['current_problem'] = f"The date {abs_offset} days {direction} is {base_date.year}/{base_date.month}/{base_date.day}, what is the date today? (The output should be in the format: 'Answer: year/month/date')"
    
    return item 
    
def verify(item):
    """
    验证答案是否正确
    参数:
        answer: 模型给出的答案
    返回:
        score: 1表示正确，0表示错误
    """

    # 标准化答案格式
    answer = str(item['action']).strip()
    correct_answer = str(item['correct_answer']).strip()
    item['score']= 1 if answer == correct_answer else 0
    return item

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    correct_answer: str
    current_problem: str
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
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)

# # 使用示例
# if __name__ == "__main__":
#     # 创建问题生成器实例
#     item = generate(224)
#     print(print_board(item))
#     item['action']=item['correct_answer']
#     item = verify(item)
#     print(f"score: {item['score']}")
#     print(item['correct_answer'])
