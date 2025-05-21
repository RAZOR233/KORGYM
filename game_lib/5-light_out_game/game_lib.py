import random
import re
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
light_out_game_prompt='''
You are a good game problem-solver, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: (0,2), (2,1)'
The game consists of a 3 by 3 grid of lights at (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1) and (2,2). '1' means the light at that position is on and '0' means the light at that position is off. When the game starts, a random number or a stored pattern of these lights is switched on. Pressing any of the lights will toggle it and the adjacent lights(up, left, right and down).For example, if the board is
000
000
000
you press the button at (1,1), the board will be
010
111
010
If the light is at the boundary of the board, it will only affect its adjacent lights. For example, if the board is
000
000
000
you press the button at (2,1), the board will be
000
010
111
The goal of this game is to switch all the lights off, preferably in as few button presses as possible. You should give you answer by a series of (a,b), which means press the light at row a and column b.You should give a series of (a,b) split by ',' to switch all the lights off.If the answer is not unique, just provide one correct answer.
Example 1:
If the board is 
000
010
111
We press the button (2,1),  which will toggle the light at (2,1) and toggle the adjacent lights (1,1), (2,0) and (2,2). The game board is
000
000
000
All the lights have been switched off. So, your answer can be 'Answer: (2,1)'.
Example 2:
If the board is 
100
011
010
First,  we press the button (0,0), which will toggle the light at (0,0) and toggle the adjacent lights (0,1) and (1,0). The game board is
010
111
010
Then, we press the button (1,1), which will toggle the light at (1,1) and toggle the adjacent lights (0,1),(1,0), (1,2) and (2,1) .The game board is
000
000
000
All the lights have been switched off. So, your answer can be 'Answer: (0,0), (1,1)'.
Example 3:
If the board is 
011
000
011
We press the button (2,2),  which will toggle the light at (2,2) and toggle the adjacent lights (2,1) and (1,2). The game board is
011
001
000
We press the button (0,2),  which will toggle the light at (0,2) and toggle the adjacent lights (0,1) and (1,2). The game board is
000
000
000
All the lights have been switched off. So, your answer can be 'Answer: (2,2) ,(0,2)'.
Board:
{board}
'''
def toggle(board, i, j):
    """切换指定位置及其相邻位置的灯状态"""
    n = len(board)
    board[i][j] ^= 1  # 切换自身
    # 切换上邻
    if i > 0:
        board[i-1][j] ^= 1
    # 切换下邻
    if i < n - 1:
        board[i+1][j] ^= 1
    # 切换左邻
    if j > 0:
        board[i][j-1] ^= 1
    # 切换右邻
    if j < n - 1:
        board[i][j+1] ^= 1

# 打印游戏板
def print_board(item):
    board=item['board']
    board_size=len(board)
    output=""
    for i in range(board_size):
        for j in range(board_size):
            output+=str(board[i][j])
            if j == board_size-1:
                output+='\n'
    return light_out_game_prompt.format(board=output)

def generate(seed):
    """生成有解的初始棋盘"""
    random.seed(seed)
    # 随机选择游戏规模n，例如3到5之间的整数
    level = random.randint(1,15)
    if level <= 5:
        n = 3
        k = level
    else:
        n = 4
        k = level-4
    # 创建全灭的初始棋盘
    board = [[0 for _ in range(n)] for _ in range(n)]
    # 生成所有可能的位置并随机选择k个不同的位置
    all_positions = [(i, j) for i in range(n) for j in range(n)]
    selected_positions = random.sample(all_positions, k)
    # 应用这些点击到初始棋盘上
    for i, j in selected_positions:
        toggle(board, i, j)
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    item['board'] = board
    item['level'] = level
    return item

def verify(item):
    """验证解题序列是否正确"""
    board=item['board']
    action_str=item['action']
    answer = [
            tuple(map(int, re.findall(r'\d+', item.strip())))  # 只提取数字并转化为 tuple
            for item in action_str.split('),') if item.strip()  # 忽略空值
        ]
    
    
    if not board:
        item['score']=0  # 空棋盘情况
    n = len(board)
    # 复制初始棋盘以避免修改原数据
    current = [row.copy() for row in board]
    # 应用所有解题步骤
    for step in answer:
        if len(step) != 2:
            item['score']=0
            return item
        i, j = step
        # 检查坐标是否合法
        if i < 0 or i >= n or j < 0 or j >= n:
            item['score']=0
            return item
        toggle(current, i, j)
    # 检查所有灯是否已灭
    for row in current:
        if any(row):
            item['score']=0
            return item
    item['score']=1
    return item

def test():
    item1={}
    item2={}
    item3={}
    # 测试用例1：正确解（点击 (0,0) 熄灭所有灯）
    item1['board'] = [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ]
    item1['action'] = [(0, 0)]
    assert verify(item1)['score'] == 1, "测试用例1失败：正确解未被接受"

    # 测试用例2：非法坐标（点击越界的 (2,2)）
    item2['board'] = [
        [1, 0],
        [0, 1]
    ]
    item2['action'] = [(2, 2)]  # 2x2 棋盘的合法坐标为 (0,0)-(1,1)
    assert verify(item2)['score'] == 0, "测试用例2失败：非法坐标未被检测"

    # 测试用例3：错误解（点击 (0,0) 后仍有灯亮）
    item3['board'] = [
		[1, 1, 1],
		[0, 1, 0],
		[0, 0, 0]
	]
    item3['action'] = [(1, 2)]
    assert verify(item3)['score'] == 0, "测试用例3失败：错误解未被检测"

    print("✅ 所有测试通过！")

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    level : int
    board : list
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
