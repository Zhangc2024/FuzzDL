results = dict()
import torch
import time
arg_1_tensor = torch.rand([5, 64, 128, 128], dtype=torch.float16)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = torch.sign(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(torch.float32)
start = time.time()
results["time_high"] = torch.sign(arg_1,)
results["time_high"] = time.time() - start

print(results)
