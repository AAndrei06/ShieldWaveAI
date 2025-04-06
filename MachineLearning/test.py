import os

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)

os.chdir(parent_path)
os.system("./startup.sh")
