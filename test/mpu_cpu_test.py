import time
import torch

TENSOR_SIZE = 20000
device = torch.device('cpu')

x = torch.rand((TENSOR_SIZE, TENSOR_SIZE), dtype=torch.float32)
y = torch.rand((TENSOR_SIZE, TENSOR_SIZE), dtype=torch.float32)
x = x.to(device)
y = y.to(device)

start_time = time.perf_counter()
result = x * y
end_time = time.perf_counter()
total_time = end_time - start_time

print(f'CPU Total execution time: {total_time} seconds')


device = torch.device('mps')
x = torch.rand((TENSOR_SIZE, TENSOR_SIZE), dtype=torch.float32)
y = torch.rand((TENSOR_SIZE, TENSOR_SIZE), dtype=torch.float32)
x = x.to(device)
y = y.to(device)

start_time = time.perf_counter()
result = x * y
end_time = time.perf_counter()
total_time = end_time - start_time

print(f'MPU Total execution time: {total_time} seconds')