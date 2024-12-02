import shutil
import os




def copy_fileinto
# 原始文件路径
source_file = "/path/to/original_file.txt"

# 目标目录
destination_dir = "/path/to/destination_folder"

# 确保目标目录存在
os.makedirs(destination_dir, exist_ok=True)

# 复制文件 50 次，并在文件名后添加编号
for i in range(50):
    # 获取原始文件的文件名和扩展名
    base_name, ext = os.path.splitext(os.path.basename(source_file))

    # 创建新的文件名
    new_file_name = f"{base_name}_{i}{ext}"
    destination_file = os.path.join(destination_dir, new_file_name)

    # 复制文件
    shutil.copy(source_file, destination_file)
    print(f"Copied: {source_file} -> {destination_file}")

print("File copying complete.")
