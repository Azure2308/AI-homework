import torch
import time
from tabulate import tabulate

# Задание 3: Сравнение производительности CPU vs CUDA (20 баллов)

# 3.1 Подготовка данных (5 баллов)
# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
# - 128 x 512 x 512
# - 256 x 256 x 256
# Заполните их случайными числами
def generate_matrices():
    shapes = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]
    return [torch.rand(shape) for shape in shapes]

# 3.2 Функции измерения времени (5 баллов)
# Создайте функции для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на GPU
# Используйте time.time() для измерения на CPU
def receive_cpu_time(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000

def receive_gpu_time(func, *args):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    func(*args)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)

def compare_operation(name, cpu_func, gpu_func, cpu_args, gpu_args):
    cpu_time = receive_cpu_time(cpu_func, *cpu_args)
    gpu_time = receive_gpu_time(gpu_func, *gpu_args) if torch.cuda.is_available() else None
    boost = cpu_time / gpu_time if gpu_time else None
    return name, cpu_time, gpu_time, boost

# 3.3 Сравнение операций (10 баллов)
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU
# 3. Вычислите ускорение
# 4. Выведите результаты в табличном виде
matrices = generate_matrices()
cudaOrCpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

for mat in matrices:
    mat_gpu = mat.to(cudaOrCpu)

    operations = [
        ("Мат. умножение", lambda x: torch.matmul(x, x.transpose(-1, -2))),
        ("Сложение", lambda x: x + x),
        ("Поэлементное умножение", lambda x: x * x),
        ("Транспонирование", lambda x: x.transpose(-1, -2)),
        ("Сумма элементов", lambda x: x.sum()),
    ]

    for op_name, op_func in operations:
        name, cpu_time, gpu_time, boost = compare_operation(
            op_name,
            op_func, op_func,
            (mat,), (mat_gpu,)
        )
        results.append((name, str(mat.shape), f"{cpu_time:.2f}",
                        f"{gpu_time:.2f}" if gpu_time else "-",
                        f"{boost:.2f}x" if boost else "-"))

headers = ["Операция", "Размер", "CPU (мс)", "GPU (мс)", "Ускорение"]
print(tabulate(results, headers=headers, tablefmt="github"))

# 3.4 Анализ результатов (5 баллов)
# Расспишу в README.md