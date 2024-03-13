results = dict()
import torch
import time
arg_1 = True
arg_2_tensor = torch.rand([1200, 1200], dtype=torch.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = torch.normal(mean=arg_1,std=arg_2,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(torch.float32)
start = time.time()
results["time_high"] = torch.normal(mean=arg_1,std=arg_2,)
results["time_high"] = time.time() - start

print(results)
