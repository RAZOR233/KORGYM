# test_app.py
import ast
from fastapi.testclient import TestClient
import requests
# 创建 TestClient 对象，用于调用 FastAPI 接口
BASE_URL = "http://localhost:8775"

def test_generate():
    """
    测试 /generate 接口，生成游戏状态 item
    """
    payload = {
        "seed": 123451,
 
    }
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    data = response.json()
    print("Generate 接口返回:")
    print(data)
    return data

def test_print_board(item):
    """
    测试 /print_board 接口，输出完整的提示信息（游戏规则、地图及行动序列）
    """
    response = requests.post(f"{BASE_URL}/print_board", json=item)
    data = response.json()
    print("\nPrint_board 接口返回:")
    print(data["board"])
    
def test_verify(item):
    """
    测试 /verify 接口
    根据生成的 game_map 和 task，利用内部 simulate 函数计算正确的最终坐标，
    将答案填入 item.answer，然后调用 /verify 接口进行验证。
    """

    # 将答案转换为字符串格式，示例格式为 "(row, col)"
    response = requests.post(f"{BASE_URL}/verify", json=item)
    item = response.json()
    print("\nVerify 接口返回:")
    print(item)
    return item
def main():
    # 依次调用测试函数
    item = test_generate()
    test_print_board(item)
    while(item['is_end']!=True):
        item['action']=input("请输入行动序列：")
        item=test_verify(item)
        test_print_board(item)
    item=test_verify(item)
    test_print_board(item)

if __name__ == "__main__":
    main()
