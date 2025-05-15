import random
import ast
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
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
game_prompt="""
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: [['row', 3], ['line', 0], ['diagonal_black', 6], ...]'

Given an  n * n  chessboard, each cell can contain either a black (B) or white (W) piece. Initially, all cells contain white pieces. You can perform the following operations:

1. Row operation (row): Turns all pieces in the selected row to white.
2. Column operation ('line'): Turns all pieces in the selected column to black.
3. Diagonal operation ('diagonal_black') (from bottom-left to top-right): Turns all pieces on the selected diagonal to black.
4. Diagonal operation ('diagonal_white') (from top-left to bottom-right): Turns all pieces on the selected diagonal to white.

Given a target pattern and a limited number of operations, your task is to achieve the target pattern starting from an all-white board.  
Output your solution as a list in the format '[[operation_name, position], ...]',e.g.'Answer: [['row', 3], ['line', 0], ['diagonal_black', 6], ...]'
Target Board:
{board}
Limited Number:
{num}
"""
# --------------------------
# 源代码逻辑（棋盘操作相关）
# --------------------------

def create_board(n):
    """生成 n*n 的初始全白棋盘，内部表示为二维列表。"""
    return [['W' for _ in range(n)] for _ in range(n)]

def apply_operation(board, op):
    """
    对棋盘 board 应用一次操作 op。
    op 为元组 (op_name, index)。
    """
    n = len(board)
    op_name, idx = op
    if op_name == "row":
        # 将第 idx 行全部设为白色
        for j in range(n):
            board[idx][j] = 'W'
    elif op_name == "line":
        # 将第 idx 列全部设为黑色
        for i in range(n):
            board[i][idx] = 'B'
    elif op_name == "diagonal_black":
        # 对 anti-diagonal：所有满足 i+j == idx 的格子设为黑色
        for i in range(n):
            for j in range(n):
                if i + j == idx:
                    board[i][j] = 'B'
    elif op_name == "diagonal_white":
        # 对 main-diagonal（左上到右下）：所有满足 i - j == idx - (n-1) 的格子设为白色
        target_diff = idx - (n - 1)
        for i in range(n):
            for j in range(n):
                if i - j == target_diff:
                    board[i][j] = 'W'
    return board

def simulate_ops(ops, n):
    """从全白棋盘出发，模拟执行 ops 序列后得到的棋盘状态。"""
    board = create_board(n)
    for op in ops:
        board = apply_operation(board, op)
    return board

def boards_equal(board1, board2):
    """判断两个棋盘状态是否相同。"""
    return all(''.join(row1) == ''.join(row2) for row1, row2 in zip(board1, board2))

def optimize_ops(ops, n):
    """
    贪心消除冗余操作：
    若移除某一步后最终棋盘状态不变，则删除该步操作，直至不能再删除为止。
    """
    target_board = simulate_ops(ops, n)
    i = 0
    while i < len(ops):
        candidate = ops[:i] + ops[i+1:]
        if boards_equal(simulate_ops(candidate, n), target_board):
            ops = candidate  # 移除第 i 步
            i = 0          # 重新从头检查
        else:
            i += 1
    return ops

def generate(seed):
    """
    根据随机种子生成目标棋盘图案以及允许的最小操作步数。
    返回值：target_map（list[str]，每个元素为棋盘的一行）、num（允许操作步数）、n（棋盘尺寸）。
    """
    random.seed(seed)
    n = 6  # 棋盘大小，可根据需要调整
    board = create_board(n)
    
    # 构造所有可能的不重复操作
    operations = []
    for r in range(n):
        operations.append(("row", r))
    for c in range(n):
        operations.append(("line", c))
    for d in range(2 * n - 1):
        operations.append(("diagonal_black", d))
    for d in range(2 * n - 1):
        operations.append(("diagonal_white", d))
    
    m = random.randint(5, min(10, len(operations)))
    random.shuffle(operations)
    chosen_ops = operations[:m]
    
    # 冗余优化
    optimized_ops = optimize_ops(chosen_ops, n)
    
    # 模拟最终棋盘状态
    final_board = simulate_ops(optimized_ops, n)
    target_map = [''.join(row) for row in final_board]
    num = len(optimized_ops)
    return target_map, num, n

def verify_ops(action, target_map, num):
    """
    验证玩家给出的操作序列 action 能否复制出目标棋盘 target_map，
    同时限制操作步数不超过 num。
    返回：1（成功）或 0（失败）。
    """
    # 操作步数超过允许数视为失败
    if len(action) > num:
        return 0
    
    n = len(target_map)
    board = create_board(n)
    performed_ops = set()
    
    for op in action:
        # 格式检查：每个操作应为长度为2的列表
        if not isinstance(op, list) or len(op) != 2:
            return 0
        op_name, op_index = op
        
        # 检查操作名称合法性
        if op_name not in ["row", "line", "diagonal_black", "diagonal_white"]:
            return 0
        
        # 检查索引范围
        if op_name in ["row", "line"]:
            if not (0 <= op_index < n):
                return 0
        elif op_name in ["diagonal_black", "diagonal_white"]:
            if not (0 <= op_index < 2 * n - 1):
                return 0
        
        # 避免重复操作
        if (op_name, op_index) in performed_ops:
            return 0
        performed_ops.add((op_name, op_index))
        
        board = apply_operation(board, (op_name, op_index))
    
    final_map = [''.join(row) for row in board]
    return 1 if final_map == target_map else 0

def board_to_str(target_map):
    """将棋盘列表转换为字符串形式，便于展示。"""
    return "\n".join(target_map)

# --------------------------
# 业务函数（供 API 调用）
# --------------------------

def generate_game_state(seed: int) -> dict:
    """
    根据种子生成完整的游戏状态，包括目标棋盘、操作步数限制及提示信息。
    """
    target_map, num, n = generate(seed)
    prompt = (
        "目标棋盘:\n" +
        "\n".join(target_map) +
        f"\n请在不超过 {num} 步内给出操作序列，每个操作形如 [操作名称, 参数]，"
        "操作名称包括：row, line, diagonal_black, diagonal_white。"
    )
    return {
        "target_map": target_map,
        "num": num,
        "n": n,
        "score": 0,
        "is_end": False,
        "action": "",
        "response": [],
        "prompt": prompt,
        "epoch": 1
    }

def verify_game_state(state: dict) -> dict:
    """
    根据玩家提供的操作序列验证是否能够复制出目标棋盘，
    同时更新 score 字段。
    """
    try:
        action = state.get('action')
        # 如果 action 为字符串，则转换为列表
        if isinstance(action, str) and action:
            action = ast.literal_eval(action)
        # 如果 action 不为字符串，那么假设它已经是列表或其他格式
        elif action is None:
            # 定义一个默认的空操作列表
            state['score'] = 0
            return state
    except Exception as e:
        state['score'] = 0
        return state
    
    state['score'] = verify_ops(action, state['target_map'], state['num'])
    return state


def get_board_str(state: dict) -> str:
    """
    返回当前目标棋盘的字符串表示。
    """
    return game_prompt.format(board=board_to_str(state['target_map']),num=state['num'])

# --------------------------
# Pydantic 模型定义
# --------------------------

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    target_map: list
    num: int
    n: int
    score: int
    is_end: bool
    action: str
    response: list
    prompt: str
    epoch: int

class BoardRequest(BaseModel):
    board: str

# --------------------------
# API 接口
# --------------------------

@app.post("/generate", response_model=GameState)
def api_generate(request: GenerateRequest):
    # 直接调用业务函数，保持 API 层简洁
    return generate_game_state(request.seed)

@app.post("/verify", response_model=GameState)
def api_verify(request: GameState):
    state = request.dict()
    return verify_game_state(state)

@app.post("/print_board", response_model=BoardRequest)
def api_print_board(request: GameState):
    return {"board": get_board_str(request.dict())}

# --------------------------
# 主程序入口
# --------------------------
if __name__ == "__main__":
    args = parse_init()
    uvicorn.run(app, host=args.host, port=args.port)