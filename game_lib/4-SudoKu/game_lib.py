import math
import ast
import random
import copy
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
size = 9
sqrt_size = int(math.sqrt(size))
difficulty_levels = {
            "easy": ((int(size * size/2) - int(size * size/10)), 0),
            "moderate": (int(size * size), int(size * size/15)),
            "difficult": (int(size * size), int(size * size/10))
        }

        # self.current_board = None
        # self.solution = None

game_prompt='''
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: happy'

{board}
'''

def print_board(item):
    """
    生成提示词
    """
    prompt = "Please solve this Sudoku puzzle. Fill in the empty cells (marked as 0) "
    prompt += "with numbers 1-9 so that each row, column, and 3x3 block contains "
    prompt += "all numbers from 1 to 9 without repetition.\n\n"
    prompt += "Please provide your solution in exactly the same format as above, "
    prompt += "i.e., a 9x9 grid where each row is a list of numbers. "
    prompt += "Example format: Answer: [[1, 2, 3, 4, 5, 6, 7, 8, 9],[4, 5, 6, 7, 8, 9, 1, 2, 3],...]\n"
    # 添加当前数独板的状态，使用代码格式
    prompt += "Current Sudoku board:\n"
    for row in item['current_board']:
        prompt += str(row) + "\n"
    return game_prompt.format(board=prompt)

def generate(seed=None):
    """
    生成数独题目
    """
    item = {
        'score': 0,
        'is_end': False,
        'response': [],
        'prompt': '',
        'action': '',
        'epoch': 1,
    }

    if seed is not None:
        random.seed(seed)
        
    # 随机选择难度
    difficulty = random.choice(["easy", "moderate", "difficult"])
    item['difficulty'] = difficulty
    print("cur difficulty: ", difficulty)
    # 生成完整的数独板
    item['solution'] = make_board()
    print("cur solution: ", item['solution'])
    # 创建题目（通过移除数字）
    item['current_board'] = [row[:] for row in item['solution']]
    item['current_board']=make_puzzle(item['current_board'],difficulty)
    
    return item

def make_board():
    """生成完整的数独解"""
    board = [[0 for _ in range(size)] for _ in range(size)]
    
    # 填充初始数独
    for i in range(size):
        for j in range(size):
            board[i][j] = int((i * sqrt_size + i//sqrt_size + j) % size) + 1
            
    # 随机打乱
    for _ in range(random.randint(8, 15)):
        board = shuffle_board(board)
        
    return board

def shuffle_board(board):
    """打乱数独板"""
    # 交换数字
    num1 = random.randint(1, size)
    num2 = random.randint(1, size)
    while num2 == num1:
        num2 = random.randint(1, size)
        
    for i in range(size):
        for j in range(size):
            if board[i][j] == num1:
                board[i][j] = num2
            elif board[i][j] == num2:
                board[i][j] = num1
                
    # 交换行
    if sqrt_size > 1:
        block = random.randint(0, sqrt_size-1)
        row1 = random.randint(0, sqrt_size-1) + block * sqrt_size
        row2 = random.randint(0, sqrt_size-1) + block * sqrt_size
        while row2 == row1:
            row2 = random.randint(0, sqrt_size-1) + block * sqrt_size
        board[row1], board[row2] = board[row2], board[row1]
    return board
        
def remove_numbers_logically(current_board, cutoff):
    """
    通过逻辑方式移除数字
    """
    removed_items = 0
    for _ in range(size * 500):
        if removed_items >= cutoff:
            break
            
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        
        # 如果该位置已经是空的，继续下一次循环
        if current_board[i][j] == 0:
            continue
            
        # 暂时保存当前数字
        temp = current_board[i][j]
        current_board[i][j] = 0
        
        # 检查在这个位置是否只有一种可能的数字
        if len(get_possible_numbers(current_board,i, j)) == 1:
            removed_items += 1
        else:
            current_board[i][j] = temp
    return current_board

def remove_numbers_randomly(current_board, cutoff):
    """
    通过随机方式移除数字
    """
    removed_items = 0
    for i in range(size):
        for j in range(size):
            if removed_items >= cutoff:
                return current_board  # 修复：返回当前棋盘而不是直接返回None
                
            if current_board[i][j] == 0:
                continue
                
            # 暂时保存当前数字
            temp = current_board[i][j]
            current_board[i][j] = 0
            
            # 创建棋盘副本进行测试
            test_board = [row[:] for row in current_board]
            
            # 检查是否有唯一解
            if solve_board(test_board):
                removed_items += 1
            else:
                # 如果没有唯一解，恢复原来的数字
                current_board[i][j] = temp
    return current_board


def get_possible_numbers(current_board, row, col):
    """
    获取指定位置所有可能的数字
    """
    possible = set(range(1, size + 1))
    
    # 检查行
    possible -= set(current_board[row])
    
    # 检查列
    possible -= set(current_board[i][col] for i in range(size))
    
    # 检查3x3宫格
    start_row = (row // sqrt_size) * sqrt_size
    start_col = (col // sqrt_size) * sqrt_size
    for i in range(start_row, start_row + sqrt_size):
        for j in range(start_col, start_col + sqrt_size):
            possible.discard(current_board[i][j])
            
    return list(possible)

def solve_board( board):
    """
    使用回溯法解数独
    """
    empty = find_empty(board)
    if not empty:
        return True
        
    row, col = empty
    
    for num in range(1, size + 1):
        if is_safe(board, row, col, num):
            board[row][col] = num
            
            if solve_board(board):
                return True
                
            board[row][col] = 0
            
    return False

def find_empty(board):
    """
    找到一个空位置
    """
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_safe( board, row, col, num):
    """
    检查在指定位置放置数字是否安全
    """
    # 检查行
    if num in board[row]:
        return False
        
    # 检查列
    if num in [board[i][col] for i in range(size)]:
        return False
        
    # 检查3x3宫格
    start_row = (row // sqrt_size) * sqrt_size
    start_col = (col // sqrt_size) * sqrt_size
    
    for i in range(start_row, start_row + sqrt_size):
        for j in range(start_col, start_col + sqrt_size):
            if board[i][j] == num:
                return False
                
    return True


def make_puzzle(current_board,difficulty):
    """根据难度移除数字创建题目"""
    logical_cutoff, random_cutoff = difficulty_levels[difficulty]
    current_board = remove_numbers_logically(current_board,logical_cutoff)
    if random_cutoff > 0:
        current_board = remove_numbers_randomly(current_board,random_cutoff)
    return current_board

def verify(item):
    """
    验证答案是否正确：
    1. 检查action中填入的数字与current_board中预设（非0）的数字是否一致；
    2. 检查每一行、每一列以及每个3x3宫格是否满足数独规则。
    如果无法转换action为列表或列表尺寸不匹配，则返回0。
    """
    try:
        # 如果item['action']为字符串，尝试转换为列表
        if isinstance(item['action'], str):
            board = ast.literal_eval(item['action'])
        else:
            board = item['action']
    except Exception as e:
        # 转换失败
        item['score'] = 0
        return item

    # 检查board是否为二维列表且行数正确
    if not isinstance(board, list) or len(board) != size:
        item['score'] = 0
        return item
    for row in board:
        if not isinstance(row, list) or len(row) != size:
            item['score'] = 0
            return item

    current_board = item['current_board']
    
    # 检查action中已填入的数字与current_board中预设的数字是否一致
    for i in range(size):
        for j in range(size):
            if current_board[i][j] != 0 and current_board[i][j] != board[i][j]:
                item['score'] = 0
                return item

    # 检查每一行是否有效
    for row in board:
        if not is_valid_unit(row):
            item['score'] = 0
            return item
            
    # 检查每一列是否有效
    for j in range(size):
        column = [board[i][j] for i in range(size)]
        if not is_valid_unit(column):
            item['score'] = 0
            return item
            
    # 检查每个3x3宫格是否有效
    for block_row in range(0, size, sqrt_size):
        for block_col in range(0, size, sqrt_size):
            block = []
            for i in range(block_row, block_row + sqrt_size):
                for j in range(block_col, block_col + sqrt_size):
                    block.append(board[i][j])
            if not is_valid_unit(block):
                item['score'] = 0
                return item
    item['score'] = 1            
    return item


def is_valid_unit(unit):
    """检查一个单元（行/列/宫格）是否有效"""
    return sorted(unit) == list(range(1, size + 1))



def get_answer(self, modelname):
    """
    获取模型的回答并解析
    """
    try:
        prompt = self.get_prompt()
        print("prompt: ", prompt)
        messages = [{"role": "user", "content": prompt}]
        response, _ = call_model(messages, modelname)
        print("reponse: ", response)
        answer_board = self._parse_answer(response)
        print("answer_board: ", answer_board)
        return answer_board
        
    except Exception as e:
        print(f"Error getting model response: {e}")
        return None

def _parse_answer(self, response):
    """
    解析模型的回答，转换为二维数组
    """
    try:
        # 提取代码块内容
        start = response.find("```") + 3
        end = response.rfind("```")
        if start == -1 or end == -1:
            raise ValueError("No code block found in response")
            
        # 获取代码块内容
        code_block = response[start:end].strip()
        
        # 解析每一行
        board = []
        for line in code_block.split('\n'):
            # 移除空行
            if not line.strip():
                continue
            # 提取数字
            try:
                # 移除方括号并分割数字
                numbers = [int(n.strip()) for n in line.strip('[]').split(',')]
                if len(numbers) != self.size:
                    raise ValueError(f"Invalid row length: {len(numbers)}")
                board.append(numbers)
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
                raise
                
        # 验证行数
        if len(board) != self.size:
            raise ValueError(f"Invalid number of rows: {len(board)}")
            
        return board
        
    except Exception as e:
        print(f"Error parsing answer: {e}")
        return None
    
def run(self, modelname):
    board = self.generate(seed=42)
    print("Generated Sudoku puzzle:")
    for row in board:
        print(row)
        
    answer_board = self.get_answer(modelname)
    score = 0
    if answer_board:
        score = generator.verify(answer_board)
        print(f"score: {score}")
    return score


class BoardRequest(BaseModel):
    board: str

class GenerateRequest(BaseModel):
    seed: int

class GameState(BaseModel):
    difficulty : str
    solution : list
    current_board : list
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

# 使用示例
# if __name__ == "__main__":
#     # generator = SudokuGenerator()
#     item = generate(2124)
#     print(f"Generated Sudoku puzzle: {item['current_board']}")
#     print(print_board(item))
#     item['action']='[[1, 5, 8, 3, 4, 7, 6, 9, 2],[6, 9, 2, 1, 5, 8, 3, 4, 7],[3, 4, 7, 6, 9, 2, 1, 5, 8],[5, 8, 3, 4, 7, 6, 9, 2, 1],[4, 7, 6, 9, 2, 1, 5, 8, 3],[9, 2, 1, 5, 8, 3, 4, 7, 6],[8, 3, 4, 7, 6, 9, 2, 1, 5],[2, 1, 5, 8, 3, 4, 7, 6, 9],[7, 6, 9, 2, 1, 5, 8, 3, 4]]'
#     score = verify(item)
#     print(f"score: {score}")
   
