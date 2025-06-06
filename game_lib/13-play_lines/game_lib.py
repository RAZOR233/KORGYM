import random
from collections import deque
import copy
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid  # 用于生成唯一标识符
from typing import Optional
import numpy as np
import ast
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
game_prompt = """
You are a good game player, I'll give you a game board and rules.
Your task is:
- First, give your answer according to the game board and rules.
- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question, e.g., "Answer: [['E','X','E',...],['E','1','1',...]...]".

Next, I will provide an n*n chessboard. On the chessboard, 'E' indicates that the element is an empty space, 'X' indicates a node that cannot be passed through, and numbers indicate nodes that need to be connected. You need to fill in the numbers on the empty spaces of the chessboard so that all identical numbers on the chessboard are connected.Moreover, the final chessboard must not have any empty spaces; every cell must be filled with a number (or remain 'X' if it's an impassable cell). Importantly, the connection for each color must form a single continuous line without branching For example, if the initial chessboard is:
E E E E E
E X E 3 E
E 3 E 1 E
E 2 E E E
1 E E 2 E
The filled chessboard could be:
2 2 2 2 2
2 X 3 3 2
2 3 3 1 2
2 2 1 1 2
1 1 1 2 2
When all the numbers on the chessboard are connected, it is considered a game victory, and you score 1 point; otherwise, if any number does not meet the connection requirement, the score will be 0.

Board:
{board}
Please output the answer in the form of a list within one line and do not break lines when outputting Answer, e.g., "Answer: [['E','X','E',...],['E','1','1',...]...]".
"""

def convert_numpy_types(item):
    if isinstance(item, dict):
        return {k: convert_numpy_types(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_numpy_types(i) for i in item]
    elif isinstance(item, tuple):
        return tuple(convert_numpy_types(i) for i in item)
    elif isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, np.floating):
        return float(item)
    elif isinstance(item, np.ndarray):
        return item.tolist()
    else:
        return item
def print_board(item):
    grid = item['puzzle_grid']
    output = ""
    for row in grid:
        output=output+''.join(str(cell) for cell in row)+'\n'
    return game_prompt.format(board = output)
def generate_endpoints(grid_size, num_colors, num_x):
    """
    随机生成谜题端点：
    在一个 grid_size x grid_size 的空网格中，每个颜色随机选取两个不重复的位置，
    并在谜题网格中标记出来（仅端点）。
    返回：谜题网格（仅端点）和端点字典 endpoints[color] = (pos1, pos2)
    """
    grid = [['E' for _ in range(grid_size)] for _ in range(grid_size)]
    endpoints = {}
    # 随机排列所有位置
    all_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(all_positions)
    # 对每个颜色选取两个位置
    for color in range(1, num_colors + 1):
        if len(all_positions) < 2:
            return None, None
        pos1 = all_positions.pop()
        # 选择第二个位置时，确保它不与第一个位置相邻
        pos2 = None
        for i in range(len(all_positions)):
            candidate = all_positions[i]
            # 检查是否与 pos1 相邻
            if abs(candidate[0] - pos1[0]) + abs(candidate[1] - pos1[1]) > 1:
                pos2 = candidate
                all_positions.pop(i)
                break
        if pos2 is None:
            return None, None
        grid[pos1[0]][pos1[1]] = color
        grid[pos2[0]][pos2[1]] = color
        endpoints[color] = (pos1, pos2)
    
    # 随机放置 num_x 个 'X'
    x_positions = random.sample(all_positions, min(num_x, len(all_positions)))
    for (x, y) in x_positions:
        grid[x][y] = 'X'
    
    return grid, endpoints

def bfs_path_solution(sol_grid, start, end, color, grid_size, allow_empty=True):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    start = tuple(start) if isinstance(start, list) else start
    end = tuple(end) if isinstance(end, list) else end
    
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    if nx < len(sol_grid) and ny < len(sol_grid[nx]):
                        cell = sol_grid[nx][ny]
                        # 若不允许空格，则只能走与color相同的格子（同时排除'X'）
                        if cell != 'X' and (cell == color or (allow_empty and cell == 'E')):
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
    return None

def compute_solution_paths(puzzle_grid, endpoints, grid_size):
    """
    根据谜题网格（仅端点）和端点字典，生成完整的解答路径。
    利用一个副本 sol_grid（初始为谜题网格），依次对各颜色求路径，
    路径求解时不允许覆盖其它颜色。
    为减少冲突，按端点间曼哈顿距离从大到小排序后求解。
    返回解答网格和 solution_paths 字典（solution_paths[color] = 路径列表）。
    """
    # 复制谜题网格作为初始解答网格
    sol_grid = copy.deepcopy(puzzle_grid)
    solution_paths = {}
    # 计算每个颜色端点的曼哈顿距离
    def manhattan(color):
        (x1, y1), (x2, y2) = endpoints[color]
        return abs(x1 - x2) + abs(y1 - y2)
    # 按距离降序排序（距离大的先连，因为较远的更难连通）
    colors = sorted(endpoints.keys(), key=lambda c: manhattan(c), reverse=True)

    for color in colors:
        start, end = endpoints[color]
        path = bfs_path_solution(sol_grid, start, end, color, grid_size)
        if path is None:
            return None, None
        solution_paths[color] = path
        # 将路径上的所有格子标记为当前颜色（注意不要覆盖其他颜色的端点）
        for (x, y) in path:
            sol_grid[x][y] = color
    return sol_grid, solution_paths

def extend_paths(sol_grid, endpoints):
    """
    从当前各颜色路径的端点出发，尝试进行强制延伸：
    - 首先扫描整个棋盘，针对每种颜色，重新计算当前端点（即与同色相邻数为1的格子）。
    - 对每个端点，检查其四邻域中空白格（'E'），如果某个端点只有唯一一个空白邻域满足：
      当该候选格填入当前颜色后，其同色邻居数恰为 1（即刚好仅与当前端点相连），则将该空白格填入当前颜色，
      同时更新该颜色的端点状态（原端点变为内部点，新填入的格子成为新的端点）。
    - 重复以上过程，直到没有任何强制延伸动作可执行。
    """
    grid_size = len(sol_grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    changed = True
    while changed:
        changed = False
        # 重新计算各颜色当前的端点（要求同色邻居数为 1）
        current_endpoints = {}
        for color in endpoints.keys():
            current_endpoints[color] = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if sol_grid[i][j] == color:
                        count = 0
                        for dx, dy in directions:
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                                if sol_grid[ni][nj] == color:
                                    count += 1
                        if count == 1:
                            current_endpoints[color].append((i, j))
        # 对每个端点尝试强制延伸
        for color, eps in current_endpoints.items():
            for ep in eps:
                i, j = ep
                candidates = []
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        if sol_grid[ni][nj] == 'E':
                            # 检查：若填入当前颜色，候选格的同色邻居数应恰为 1（仅与当前端点相连）
                            count_same = 0
                            for ddx, ddy in directions:
                                nni, nnj = ni + ddx, nj + ddy
                                if 0 <= nni < grid_size and 0 <= nnj < grid_size:
                                    if sol_grid[nni][nnj] == color:
                                        count_same += 1
                            if count_same == 1:
                                candidates.append((ni, nj))
                # 仅当候选唯一时，执行延伸
                if len(candidates) == 1:
                    sol_grid[candidates[0][0]][candidates[0][1]] = color
                    changed = True
    return sol_grid

def generate(seed):
    """
    修改后的 generate 函数：
      1. 随机生成 grid_size、num_colors 及障碍数（'X'），这里建议适当降低障碍数以提高成功率。
      2. 利用 generate_endpoints 生成仅含端点的谜题网格。
      3. 调用 compute_solution_paths 对各颜色利用 BFS 求得初步连通路径（这一步保证每条路径本身是一条单线）。
      4. 调用 extend_paths 对初步路径进行强制延伸，填充剩余空格，使棋盘尽可能被路径覆盖。
      5. 若延伸后棋盘仍存在空格，或整体不满足 check_no_branching（即每种颜色仍需满足端点同色邻居为 1，中间格子为 2），则本次生成失败，重试新的参数。
    """
    random.seed(seed)
    count=0
    while True:  # 外层循环，确保在生成失败时重新生成参数
        grid_size = random.randint(5, 10)      # 棋盘尺寸 5x5 到 10x10
        num_colors = random.randint(3,5)        # 颜色数量 5 到 8
        num_x = random.randint(1, 3)             # 为提高延伸成功率，适当减少障碍数量
        max_attempts = random.randint(5000, 10000)  # 最大尝试次数

        attempts = 0
        while attempts < max_attempts:
            puzzle_grid, endpoints = generate_endpoints(grid_size, num_colors, num_x)
            if puzzle_grid is None:
                attempts += 1
                continue
            sol_grid, solution_paths = compute_solution_paths(puzzle_grid, endpoints, grid_size)
            if sol_grid is None:
                attempts += 1
                continue
            # 利用强制延伸填充剩余空格
            sol_grid = extend_paths(sol_grid, endpoints)
            # 如果延伸后仍有空格，则视为失败
            if any('E' in row for row in sol_grid):
                attempts += 1
                continue
            # 整体检查：确保每种颜色的路径为单线（端点同色邻居为1，其余为2）
            if not check_no_branching(sol_grid, endpoints):
                attempts += 1
                continue

            item = {
                'puzzle_grid': puzzle_grid,
                'endpoints': endpoints,
                'score': 1,      # 生成成功
                'is_end': False,
                'response': [],
                'prompt': '',
                'action': "",  # 完整的解答棋盘存入 action 字段
                'epoch': 1,
                'grid_size': grid_size,
            }
            return item
        print(f"Retrying {count}")
        count+=1






# 新增辅助函数：检查每个颜色的连接是否为单一不分叉的路径
def check_no_branching(sol_grid, endpoints):
    """
    对于每种颜色，遍历其所有出现的格子，
    检查：
      - 对于起点和终点，必须只有1个相邻同色格子；
      - 对于其它格子，必须正好有2个相邻同色格子。
    如果存在不满足条件的情况，返回 False，否则返回 True。
    """
    grid_size = len(sol_grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for color, (start, end) in endpoints.items():
        # 将起点、终点转换为元组（若有需要）
        if isinstance(start, list):
            start = tuple(start)
        if isinstance(end, list):
            end = tuple(end)
        # 收集该颜色在棋盘中的所有格子位置
        cells = [(i, j) for i in range(grid_size) for j in range(grid_size) if sol_grid[i][j] == color]
        # 对每个格子统计相邻同色个数
        endpoint_count = 0
        for i, j in cells:
            count = 0
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    if sol_grid[ni][nj] == color:
                        count += 1
            # 如果该位置是起点或终点，要求仅有1个同色邻居
            if (i, j) == start or (i, j) == end:
                if count != 1:
                    return False
                endpoint_count += 1
            else:
                # 中间的格子必须正好有2个同色邻居
                if count != 2:
                    return False
        # 检查该颜色区域中必须正好有2个端点（起点与终点）
        if endpoint_count != 2:
            return False
    return True

# 修改后的 verify 函数，增加了对不分叉要求的验证
def verify(item):
    # 确保 action 是列表
    if isinstance(item['action'], str):
        try:
            sol_grid = ast.literal_eval(item['action'])
        except (ValueError, SyntaxError):
            item['score'] = 0
            item['is_end'] = True
            return item
    else:
        sol_grid = item['action']
    
    if not isinstance(sol_grid, list) or not all(isinstance(row, list) for row in sol_grid):
        item['score'] = 0
        item['is_end'] = True
        return item
    
    endpoints = item['endpoints']
    grid_size = item['grid_size']
    
    for color in endpoints:
        if isinstance(endpoints[color], list):
            endpoints[color] = tuple(map(tuple, endpoints[color]))
    
    # 先验证各颜色连通
    for color, (start, end) in endpoints.items():
        # 验证时不允许走空格
        path = bfs_path_solution(sol_grid, start, end, color, grid_size, allow_empty=False)
        if path is None:
            item['score'] = 0
            item['is_end'] = True
            return item

    # 验证棋盘内不允许有空格
    for row in sol_grid:
        if 'E' in row:
            item['score'] = 0
            item['is_end'] = True
            return item

    # 新增验证：不允许分叉，每个颜色必须是一条单线连接
    if not check_no_branching(sol_grid, endpoints):
        item['score'] = 0
        item['is_end'] = True
        return item

    item['score'] = 1
    item['is_end'] = True
    return item




def print_grid(grid):
    """ 打印二维网格 """
    for row in grid:
        print(' '.join(str(cell) for cell in row))

def print_solution(sol_grid):
    """ 打印解答网格 """
    print("Solution Grid:")
    print_grid(sol_grid)

def main():
    count=0
    for i in range(2):
        item = generate(i)
        if item['puzzle_grid'] is None:
            count+=1
            print("Failed to generate a solvable puzzle after multiple attempts.")
        else:
            print("Puzzle Grid (Endpoints Only):")
            print_grid(item['puzzle_grid'])
            print()
            print_solution(item['action'])
            print("\nTesting solution...")
            item = verify(item)
            if item['score'] == 1:
                print("Test passed: All paths are valid.")
            else:
                print("Test failed: Some paths are invalid.")
            print(print_board(item))
    print(count)
# if __name__ == "__main__":
#     main()
# --- 定义请求和响应数据模型 ---

class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    puzzle_grid: list
    endpoints: dict
    grid_size: int
    score: float
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

# --- API 接口 ---

# 生成游戏板内容
@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    state = request.dict()
    board_output = print_board(state)
    return {"board": board_output}

# 生成初始游戏状态
@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    game_state = generate(request.seed)
    # 转换 NumPy 数据类型
    game_state = convert_numpy_types(game_state)
    return game_state

# 根据动作更新游戏状态
@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    # 转换 endpoints 中的值为元组
    if 'endpoints' in state:
        endpoints = state['endpoints']
        for color in endpoints:
            if isinstance(endpoints[color], list):  # 如果值是列表
                endpoints[color] = tuple(map(tuple, endpoints[color]))  # 将列表转换为元组
        state['endpoints'] = endpoints
    updated_state = verify(state)
    # 转换 NumPy 数据类型后返回
    updated_state = convert_numpy_types(updated_state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)

    # main()