results = dict()
import torch
import time
arg_1 = 32
start = time.time()
results["time_low"] = torch.set_num_threads(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = torch.set_num_threads(arg_1,)
results["time_high"] = time.time() - start

print(results)
