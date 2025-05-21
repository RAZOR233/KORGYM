import random
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
import ast
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
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g."Answer: [['R',3,2], ['U',0,2], ...]".

Given an 'n*n' maze containing empty spaces ('E'), a protagonist ('P'), walls ('X'), and numbered waypoints ('digits') that must be visited in sequence. You are provided with an initial player movement direction ('up/down/left/right') and a series of player actions ('U/D/L/R') along with their respective counts. The player needs to produce an action sequence such that the protagonist changes direction automatically when reaching each waypoint, ensuring that waypoints are visited sequentially. The action sequence must trigger the waypoints strictly in order; if the second waypoint isn't triggered, subsequent waypoints will not be triggered even if visited.The coordinates in the top left corner are [0,0].

Please output the sequence of actions and corresponding trigger positions in the following format:
'Answer: [['R',3,2], ['U',0,2], ...]'
Maze Board:
{board_str}
Current Direction:{initial_direction}
Device Actions:{device_actions}
"""
def generate(seed: int):
    """
    根据种子生成一个 15x15 的迷宫，其中包含：
      - 主角起点（P），放在与初始方向对应的边上；
      - 固定 3 个必经途径点，分别标记为 "1", "2", "3"；
      - 空白区域（E）和墙壁（X）。
    同时随机确定初始方向，并根据曼哈顿路径规划生成一条完整解路，
    并记录在解路过程中真正需要改变方向的设备动作（格式：[command, row, col]）。
    """
    random.seed(seed)
    n = random.randint(7,15)
    directions = ["up", "down", "left", "right"]
    initial_direction = random.choice(directions)
    
    # 根据初始方向选择 P 所在边（保证初始运动方向有效）
    if initial_direction == "right":
        P = (random.randint(0, n-1), 0)       # 左边缘
    elif initial_direction == "left":
        P = (random.randint(0, n-1), n-1)       # 右边缘
    elif initial_direction == "down":
        P = (0, random.randint(0, n-1))         # 上边缘
    else:  # up
        P = (n-1, random.randint(0, n-1))       # 下边缘

    # 固定途径点数量（例如 3 个）
    num_waypoints = 3
    waypoints = []
    for _ in range(num_waypoints):
        pos = (random.randint(0, n-1), random.randint(0, n-1))
        while pos == P or pos in waypoints:
            pos = (random.randint(0, n-1), random.randint(0, n-1))
        waypoints.append(pos)
    
    # --- 曼哈顿路径规划 ---
    def manhattan_path(start, end, order):
        """
        生成从 start 到 end 的曼哈顿路径，
        order 参数可选 "horizontal_first" 或 "vertical_first"
        """
        path = [start]
        r0, c0 = start
        r1, c1 = end
        if order == "horizontal_first":
            # 水平移动
            if c1 > c0:
                for c in range(c0+1, c1+1):
                    path.append((r0, c))
            elif c1 < c0:
                for c in range(c0-1, c1-1, -1):
                    path.append((r0, c))
            # 垂直移动
            if r1 > r0:
                for r in range(r0+1, r1+1):
                    path.append((r, c1))
            elif r1 < r0:
                for r in range(r0-1, r1-1, -1):
                    path.append((r, c1))
        else:  # vertical_first
            if r1 > r0:
                for r in range(r0+1, r1+1):
                    path.append((r, c0))
            elif r1 < r0:
                for r in range(r0-1, r1-1, -1):
                    path.append((r, c0))
            if c1 > c0:
                for c in range(c0+1, c1+1):
                    path.append((r1, c))
            elif c1 < c0:
                for c in range(c0-1, c1-1, -1):
                    path.append((r1, c))
        return path

    def get_direction(a, b):
        """返回从 a 到 b 的移动方向"""
        r0, c0 = a
        r1, c1 = b
        if r1 == r0 and c1 == c0 + 1:
            return "right"
        elif r1 == r0 and c1 == c0 - 1:
            return "left"
        elif r1 == r0 + 1 and c1 == c0:
            return "down"
        elif r1 == r0 - 1 and c1 == c0:
            return "up"
        return None

    full_path = []
    current = P
    current_dir = initial_direction
    # 分段连接 P 与各途径点的路径
    for wp in waypoints:
        path_h = manhattan_path(current, wp, "horizontal_first")
        path_v = manhattan_path(current, wp, "vertical_first")
        def count_turns(path, init_dir):
            cnt = 0
            d = init_dir
            for i in range(len(path)-1):
                nd = get_direction(path[i], path[i+1])
                if nd != d:
                    cnt += 1
                    d = nd
            return cnt
        cnt_h = count_turns(path_h, current_dir)
        cnt_v = count_turns(path_v, current_dir)
        if cnt_h <= cnt_v:
            chosen_path = path_h
            if len(chosen_path) >= 2:
                current_dir = get_direction(chosen_path[-2], chosen_path[-1])
        else:
            chosen_path = path_v
            if len(chosen_path) >= 2:
                current_dir = get_direction(chosen_path[-2], chosen_path[-1])
        # 拼接路径时避免重复当前点
        if full_path and full_path[-1] == chosen_path[0]:
            full_path.extend(chosen_path[1:])
        else:
            full_path.extend(chosen_path)
        current = wp

    # --- 生成最小设备列表（仅记录转向时刻） ---
    mapping = {"up": "U", "down": "D", "left": "L", "right": "R"}
    device_actions = []
    # 检查第一步是否与初始方向一致
    if len(full_path) >= 2:
        first_move = get_direction(full_path[0], full_path[1])
        if first_move != initial_direction:
            # 起点放置设备使方向调整为 first_move
            device_actions.append([mapping[first_move], full_path[0][0], full_path[0][1]])
            current_dir = first_move
        else:
            current_dir = initial_direction
    # 遍历 full_path 找出转向时刻，记录设备动作
    for i in range(1, len(full_path)-1):
        d_prev = get_direction(full_path[i-1], full_path[i])
        d_next = get_direction(full_path[i], full_path[i+1])
        if d_next != d_prev:
            device_actions.append([mapping[d_next], full_path[i][0], full_path[i][1]])
            current_dir = d_next

    # --- 构造迷宫 ---
    # 初始迷宫全为墙壁 "X"
    maze = [["X" for _ in range(n)] for _ in range(n)]
    # 将 full_path 上的单元设为通路 "E"
    for (r, c) in full_path:
        maze[r][c] = "E"
    # 标记起点
    maze[P[0]][P[1]] = "P"
    # 按顺序标记途径点
    for i, wp in enumerate(waypoints, start=1):
        r, c = wp
        maze[r][c] = str(i)
    # 对非解路区域随机填充，保证解路不被破坏
    for i in range(n):
        for j in range(n):
            if (i, j) not in full_path:
                maze[i][j] = "E" if random.random() < 0.4 else "X"
    
    item = {
        "maze": maze,
        "initial_direction": initial_direction,
        "device_actions": device_actions,
        "score": 0,
        "is_end": False,
        "action": '',     # 用户提交的设备动作序列，格式为 [[command, row, col], ...]
        "response": [],
        "prompt": "",     # 可由 print_board 接口生成
        "epoch": 1,
        "n": n
    }
    return item

def verify_actions(actions, maze, initial_direction):
    """
    根据用户给定的设备动作序列模拟主角移动：
      - 主角从起点 P 出发，初始方向为 initial_direction；
      - 当到达设备动作预期位置时，自动触发设备指令，更新移动方向；
      - 主角需依次经过迷宫中标记的数字（必经途径点），否则视为失败。
    返回 1 表示验证成功，0 表示失败。
    """
    n = len(maze)
    device_index = 0
    mapping = {"U": "up", "D": "down", "L": "left", "R": "right"}
    
    # 寻找起点 P
    start = None
    for i in range(n):
        for j in range(n):
            if maze[i][j] == "P":
                start = (i, j)
                break
        if start is not None:
            break
    if start is None:
        return 0
    
    pos = start
    direction = initial_direction
    expected = 1  # 下一个应经过的途径点数字
    max_waypoint = 0
    for i in range(n):
        for j in range(n):
            if maze[i][j].isdigit():
                max_waypoint = max(max_waypoint, int(maze[i][j]))
    
    max_steps = 1000
    steps = 0
    while steps < max_steps:
        # 如果当前位置与预期设备放置位置匹配，则触发设备更新方向
        if device_index < len(actions):
            exp_act = actions[device_index]  # 格式 [command, row, col]
            if pos == (exp_act[1], exp_act[2]):
                direction = mapping[exp_act[0]]
                device_index += 1
        # 检查是否到达必经途径点
        cell_val = maze[pos[0]][pos[1]]
        if cell_val.isdigit() and int(cell_val) == expected:
            expected += 1
            if expected > max_waypoint and max_waypoint > 0:
                return 1
        # 按当前方向移动
        r, c = pos
        if direction == "up":
            nr, nc = r - 1, c
        elif direction == "down":
            nr, nc = r + 1, c
        elif direction == "left":
            nr, nc = r, c - 1
        elif direction == "right":
            nr, nc = r, c + 1
        else:
            return 0
        if not (0 <= nr < n and 0 <= nc < n) or maze[nr][nc] == "X":
            return 0
        pos = (nr, nc)
        steps += 1
    return 0

def verify_item(item):
    """
    根据 item 中的动作序列验证解路是否有效，
    并更新 item['score']（1 为成功，0 为失败）。
    如果发生解析或验证时异常，则 score=0 并返回 item。
    """
    try:
        # 如果 action 是字符串，就用 literal_eval 把它变成列表
        if isinstance(item['action'], str):
            actions = ast.literal_eval(item['action'])
        else:
            actions = item['action']

        maze = item.get("maze")
        initial_direction = item.get("initial_direction")

        # 这一行可能会抛出 IndexError、ValueError 等
        score = verify_actions(actions, maze, initial_direction)

    except Exception:
        # 任何异常都当作失败
        item["score"] = 0
        return item

    # 验证成功或验证函数正常返回
    item["score"] = score
    return item


def print_board(item):
    """
    将 item 中的迷宫 board 生成文本形式，
    每行以空格分隔。
    """
    maze = item.get("maze", [])
    board_str = ""
    for row in maze:
        board_str += " ".join(row) + "\n"
    initial_direction = item['initial_direction']
    device_actions = [k[0] for k in item['device_actions']]
    return game_prompt.format(board_str=board_str,initial_direction=initial_direction,device_actions=device_actions)

# ----- FastAPI 接口定义 -----

class GenerateRequest(BaseModel):
    seed: int

class BoardRequest(BaseModel):
    board: str

class GameState(BaseModel):
    maze: list
    initial_direction: str
    device_actions: list
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int
    n: int

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    """
    根据给定种子生成初始迷宫状态
    """
    item = generate(request.seed)
    # 同时生成迷宫文本提示
    return item

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    """
    根据当前状态返回迷宫的文本表示
    """
    board_output = print_board(request.dict())
    return {"board": board_output}

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    """
    根据用户提交的动作序列验证解路是否正确，
    并返回更新后的状态信息
    """
    state = request.dict()
    updated_state = verify_item(state)
    return updated_state

if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)