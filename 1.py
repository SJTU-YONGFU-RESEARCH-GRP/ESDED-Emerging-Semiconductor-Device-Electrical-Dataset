import os
import re

def count_unique_xxxx(folder_path):
    """
    统计文件夹中不同xxxx的数量
    
    参数:
        folder_path (str): 文件夹路径
        
    返回:
        int: 不同xxxx的数量
        set: 所有唯一的xxxx集合
    """
    xxxx_set = set()
    pattern = re.compile(r'^(\d+)_[0-9A-Za-z]+$')  # 匹配xxxx_yyz格式
    
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            xxxx = match.group(1)
            xxxx_set.add(xxxx)
    
    return len(xxxx_set), xxxx_set

# 使用示例
if __name__ == "__main__":
    folder_path = input("请输入文件夹路径: ")
    count, xxxx_set = count_unique_xxxx(folder_path)
    print(f"共有 {count} 种不同的DOI")
    print("具体的DOI有:", sorted(xxxx_set))