import os

def get_size(start_path):
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except:
                        pass
    except Exception:
        pass
    return total_size / (1024 * 1024 * 1024)

paths = [
    r'C:\Users\Amin-PC\.cache\huggingface',
    r'C:\Users\Amin-PC\.insightface',
    r'C:\Users\Amin-PC\.deepface',
    r'C:\Users\Amin-PC\AppData\Local\pip\cache',
    r'C:\Users\Amin-PC\AppData\Local\Ultralytics',
    r'C:\Users\Amin-PC\.conda',
    r'C:\Users\Amin-PC\miniconda3',
    r'C:\Users\Amin-PC\Anaconda3',
    r'C:\Users\Amin-PC\.virtualenvs',
    r'C:\Users\Amin-PC\Desktop\Montiring system\.venv',
    r'C:\Users\Amin-PC\Desktop\Montiring system\venv',
]

print("Checking common Cache and Env directories:")
for p in paths:
    if os.path.exists(p):
        size_gb = get_size(p)
        print(f"{p} : {size_gb:.2f} GB")
