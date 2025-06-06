import random
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
problem_2048_prompt='''\
You are a good game problem-solver, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: LEFT'
Rules:The game is played on a 4x4 grid, with each tile containing a number that is a power of 2 (e.g., 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048). Your goal is to combine the tiles to have more scores. The game ends when there are no more valid moves, or when you achieve the 2048 tile.In the game board, 0 means empty tile and | means the delimiter between tiles. At the beginning of the game, two tiles with the number 2 or 4 will appear randomly on the grid. You can swipe left, right, up, or down to move all tiles in that direction. All tiles will shift to the edge of the grid, and any empty spaces will be filled by a new tile (2 or 4).When two tiles of the same number touch, they will merge into one tile with the sum of those numbers and you will get the score of the new tiles. For example, two tiles with the number 2 will merge to form a 4. After merging, the new tile will not combine again in the same move. You lose the game if the grid is full, and no valid moves are left. A valid move is when two adjacent tiles are the same or there is an empty space to move a tile into. Keep in mind that combining tiles strategically is key. Try to keep the larger tiles in a corner and work towards merging smaller tiles to get higher scores.Remember, the game will end after the 100th epoch.
For example,if the Game board is
0|0|4|0
0|2|0|8
0|0|4|0
0|0|0|2
and the answer is DOWN

the next state of Game board will be
0|0|0|0
0|0|0|0
0|0|0|8
0|2|8|2
and since the two '4' merge into '8',so you will get 8 score
Game board:
{board}
Current epoch: {epoch}
The answer you give should be one of 'LEFT', 'RIGHT', 'UP' and 'DOWN'
'''

# 新砖块生成机制：
# 根据当前棋盘上的最大砖块，允许的新砖块取值为2的幂，
# 范围为2 ~ (最大砖块 // 2)，若当前最大砖块小于4，则只能生成2。
def get_new_tile_value(board):
    max_tile = max(max(row) for row in board)
    # 当棋盘还没有砖块或最大砖块小于4时，默认返回2
    if max_tile < 4:
        return 2
    allowed = []
    v = 2
    # 允许的最大值为当前最大砖块的一半
    while v <= max_tile // 2:
        allowed.append(v)
        v *= 2
    if not allowed:
        allowed = [2]
    return random.choice(allowed)

# 打印游戏板
def print_board(item):
    board = item['board']
    board_size = len(board)
    output = ""
    for i in range(board_size):
        for j in range(board_size):
            output += str(board[i][j])
            if j != board_size - 1:
                output += '|'
            else:
                output += '\n'
    prompt = problem_2048_prompt.format(board=output, epoch=item['epoch'])
    return prompt

def generate(seed: int):
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }
    random.seed(seed)
    board = [[0] * 4 for _ in range(4)]
    positions = random.sample(range(16), 2)
    for pos in positions:
        # 使用新的砖块生成机制
        board[pos // 4][pos % 4] = get_new_tile_value(board)
    item['board'] = board
    item['score'] = 0
    item['is_end'] = False
    return item

def compress(board):
    new_board = [[0] * 4 for _ in range(4)]
    score = 0
    for i in range(4):
        filtered = [num for num in board[i] if num != 0]
        new_line = []
        skip = False
        for j in range(len(filtered)):
            if skip:
                skip = False
                continue
            if j < len(filtered) - 1 and filtered[j] == filtered[j + 1]:
                new_line.append(filtered[j] * 2)
                score += filtered[j] * 2
                skip = True
            else:
                new_line.append(filtered[j])
        while len(new_line) < 4:
            new_line.append(0)
        new_board[i] = new_line
    return new_board, score

def rotate(board, times=1):
    for _ in range(times):
        board = [list(row) for row in zip(*board[::-1])]
    return board

def can_move(board):
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                return True
            if j < 3 and board[i][j] == board[i][j + 1]:
                return True
            if i < 3 and board[i][j] == board[i + 1][j]:
                return True
    return False

def verify(item):
    temp_board = [row[:] for row in item['board']]
    score = item['score']
    move = item['action'].strip().lower()
    if move not in ['up', 'down', 'left', 'right']:
        move = random.choice(['up', 'down', 'left', 'right'])

    rotations = {'down': 1, 'up': 3, 'left': 0, 'right': 2}
    temp_board = rotate(temp_board, rotations[move])
    temp_board, gained_score = compress(temp_board)
    temp_board = rotate(temp_board, (4 - rotations[move]) % 4)

    score += gained_score

    empty_positions = [(i, j) for i in range(4) for j in range(4) if temp_board[i][j] == 0]
    if empty_positions:
        i, j = random.choice(empty_positions)
        # 使用新的砖块生成机制
        temp_board[i][j] = get_new_tile_value(temp_board)
    item['board'] = temp_board
    item['score'] = score
    item['epoch'] += 1
    return item

def play_game(seed):
    item = generate(seed)
    while True:
        print("\nCurrent Board:")
        print(print_board(item))
        print("Score:", item['score'])
        if item['is_end']:
            print("Game Over! No more valid moves. Final Score:", item['score'])
            break
        move = input("Enter move (up/down/left/right or 'quit' to exit): ")
        if move == 'quit':
            print("Game Over! Final Score:", item['score'])
            break
        else:
            item['action'] = move
        try:
            item = verify(item)
        except ValueError:
            print("Invalid move. Try again.")
            continue

# --- 定义请求和响应数据模型 ---

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    board: list
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

# --- API 接口 ---

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
    state = request.dict()
    updated_state = verify(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)
