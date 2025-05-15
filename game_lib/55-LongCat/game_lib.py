import random
from collections import deque
import copy
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ast
import argparse

def parse_init():
    """
    定义并解析命令行参数，用于服务部署地址与端口的配置。
    """
    parser = argparse.ArgumentParser(description="Data creation utility")
    parser.add_argument('-p', '--port', type=int, default=8775, help='服务部署端口')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='服务部署地址')
    args = parser.parse_args()
    return args
app = FastAPI()
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: ['left', 'down', 'right', 'up', 'left']'.

Next, I will provide an n × n board containing a cat ('C'), empty spaces ('E'), and walls ('X'). You need to control the cat's movement by entering directions: up, down, left, or right. The cat moves from its initial position, sliding continuously in the chosen direction until hitting a wall. All empty spaces ('E') traversed along the path will turn into walls ('X'). The game is won when all empty spaces have been filled. Please output your solution as a list containing directions ('up', 'left', 'right', 'down'), for example:  
'Answer: ['left', 'down', 'right', 'up', 'left']'
Board:
{board}
"""
# ================================
# 原始辅助函数（地图生成与验证逻辑）
# ================================

def is_solvable(game_map):
    """
    使用 DFS 搜索判断游戏地图是否有解。
    游戏规则：猫从初始位置出发，每次沿某一方向滑动，直至遇到墙壁，
    路径上所有空格（'E'）将变为墙壁（'X'）。当所有空格都被填充时游戏胜利。
    """
    rows = len(game_map)
    cols = len(game_map[0])
    cat_pos = None
    board = [row.copy() for row in game_map]
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'C':
                cat_pos = (r, c)
                board[r][c] = 'X'
                break
        if cat_pos:
            break
    if not cat_pos:
        return False

    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    def board_to_key(board, cat_pos):
        return (cat_pos, tuple(tuple(row) for row in board))

    visited = {}

    def dfs(board, cat_pos):
        key = board_to_key(board, cat_pos)
        if key in visited:
            return visited[key]
        # 若所有空格均已填充，则找到解
        if all(cell != 'E' for row in board for cell in row):
            visited[key] = True
            return True
        for dr, dc in directions.values():
            r, c = cat_pos
            path = []
            while True:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                if board[nr][nc] == 'X':
                    break
                if board[nr][nc] == 'E':
                    path.append((nr, nc))
                r, c = nr, nc
            if not path:
                continue
            new_board = [list(row) for row in board]
            for pr, pc in path:
                new_board[pr][pc] = 'X'
            new_cat_pos = path[-1]
            if dfs(new_board, new_cat_pos):
                visited[key] = True
                return True
        visited[key] = False
        return False

    return dfs(board, cat_pos)

def init_map(rows, cols):
    """初始化地图：边缘为 X，内部为 E"""
    return [['X' if r == 0 or r == rows - 1 or c == 0 or c == cols - 1 else 'E'
             for c in range(cols)] for r in range(rows)]

def is_valid(r, c, rows, cols):
    """检查坐标是否在地图范围内"""
    return 0 <= r < rows and 0 <= c < cols

def get_neighbors(game_map, r, c, cell_type):
    """获取指定类型的相邻单元格（上下左右）"""
    rows = len(game_map)
    cols = len(game_map[0])
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if is_valid(nr, nc, rows, cols) and game_map[nr][nc] == cell_type:
            neighbors.append((nr, nc))
    return neighbors

def bfs_connectivity(game_map, start):
    """使用 BFS 检查连通性，返回从 start 出发可到达的所有 'E' 单元格"""
    rows = len(game_map)
    cols = len(game_map[0])
    visited = set()
    q = deque([start])
    visited.add(start)

    while q:
        r, c = q.popleft()
        for nr, nc in get_neighbors(game_map, r, c, 'E'):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited

def add_random_walls(game_map, rows, cols):
    """
    随机添加少量内部墙壁，确保剩余 'E' 连通。
    添加约内部单元格数量的 1/5 个墙壁。
    """
    internal_cells = [(r, c) for r in range(1, rows-1)
                      for c in range(1, cols-1) if game_map[r][c] == 'E']
    num_walls = len(internal_cells) // 5

    for _ in range(num_walls):
        r, c = random.choice(internal_cells)
        original = game_map[r][c]
        game_map[r][c] = 'X'
        e_cells = [(r, c) for r in range(rows) for c in range(cols) if game_map[r][c] == 'E']
        if e_cells:
            visited = bfs_connectivity(game_map, e_cells[0])
            if len(visited) != len(e_cells):
                game_map[r][c] = original

def place_cat(game_map, rows, cols):
    """
    放置猫咪到叶子节点（只有一个相邻 'E'）或随机一个 'E' 单元格
    """
    leaves = []
    all_e = []
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if game_map[r][c] == 'E':
                all_e.append((r, c))
                if len(get_neighbors(game_map, r, c, 'E')) == 1:
                    leaves.append((r, c))
    if leaves:
        cat_r, cat_c = random.choice(leaves)
    else:
        if not all_e:
            raise RuntimeError("没有剩余的空格用于放置猫咪")
        cat_r, cat_c = random.choice(all_e)
    game_map[cat_r][cat_c] = 'C'

def generate_map(seed: int):
    """
    生成完整地图：不断生成-检验，直至生成一个保证有解的地图。
    返回的 item 中包含地图信息及其他游戏状态数据。
    """
    random.seed(seed)
    item = {
        "score": 0,
        "is_end": False,
        "action": "",       # 用户提交的移动序列，格式为列表，如 ['right', 'down', ...]
        "response": [],
        "prompt": "",
        "epoch": 1,
    }
    rows = random.randint(5, 10)
    cols = random.randint(5, 10)
    item['row_num'] = rows
    item['col_num'] = cols 
    
    attempt = 0
    while True:
        attempt += 1
        game_map = init_map(rows, cols)
        add_random_walls(game_map, rows, cols)
        place_cat(game_map, rows, cols)
        if is_solvable(game_map):
            item['game_map'] = game_map
            break
    return item

def verify(actions, game_map):
    """
    验证移动序列是否能填充所有空格：
      - 猫从初始位置出发，沿给定方向滑动直到遇到墙壁，
        路径上所有 'E' 变为 'X'（猫移动后原位置也变为 'X'）。
      - 最终所有 'E' 均被填充，则返回 1，否则返回 0。
    """
    current_map = [row.copy() for row in game_map]
    rows = len(current_map)
    cols = len(current_map[0]) if rows > 0 else 0

    # 寻找猫的初始位置
    cat_pos = None
    for r in range(rows):
        for c in range(cols):
            if current_map[r][c] == 'C':
                cat_pos = (r, c)
                current_map[r][c] = 'X'  # 猫移动后原位置变为 X
                break
        if cat_pos:
            break

    if not cat_pos:
        return 0

    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    for move in actions:
        move = move.lower()
        if move not in directions:
            continue
        dr, dc = directions[move]
        current_r, current_c = cat_pos
        path = []
        while True:
            next_r = current_r + dr
            next_c = current_c + dc
            if not (0 <= next_r < rows and 0 <= next_c < cols):
                break
            if current_map[next_r][next_c] == 'X':
                break
            if current_map[next_r][next_c] == 'E':
                path.append((next_r, next_c))
            current_r, current_c = next_r, next_c
        if not path:
            continue  # 无效移动
        for r, c in path:
            current_map[r][c] = 'X'
        cat_pos = path[-1]
    for row in current_map:
        if 'E' in row:
            return 0
    return 1

def verify_game(item):
    """
    根据 item 中的 action 和 game_map 验证移动序列能否填充所有空格，
    并更新 item 中的 score 字段。
    """
    actions = item.get("action")
    if isinstance(actions, str):
        try:
            actions = ast.literal_eval(actions)
        except Exception as e:
            item["score"] = 0
            return item
    game_map = item.get("game_map")
    score = verify(actions, game_map)
    item["score"] = score
    return item

def print_board(item):
    """
    将 game_map 转换为字符串格式，每一行以空格分隔、换行连接
    """
    game_map = item.get("game_map", [])
    board_str = "\n".join([" ".join(row) for row in game_map])
    return game_prompt.format(board=board_str)

# ================================
# FastAPI 接口及数据模型
# ================================

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    game_map: list
    row_num: int
    col_num: int
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate_map(request.seed)
    return game_state

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    updated_state = verify_game(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)