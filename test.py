import os
cur_path = os.getcwd()  # 当前工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
print(cur_path)
print(script_dir)
