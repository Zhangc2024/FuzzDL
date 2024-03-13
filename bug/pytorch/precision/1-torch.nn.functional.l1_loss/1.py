results = dict()
import torch
import time
arg_1_tensor = torch.rand([128, 1184, 4, 4], dtype=torch.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(-1,128,[1, 4], dtype=torch.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = "mean"
start = time.time()
results["time_low"] = torch.nn.functional.l1_loss(arg_1,arg_2,reduction=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(torch.float32)
arg_2 = arg_2_tensor.clone().type(torch.int64)
start = time.time()
results["time_high"] = torch.nn.functional.l1_loss(arg_1,arg_2,reduction=arg_3,)
results["time_high"] = time.time() - start

print(results)
