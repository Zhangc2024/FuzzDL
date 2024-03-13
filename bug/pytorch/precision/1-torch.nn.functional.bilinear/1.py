results = dict()
import torch
import time
arg_1_tensor = torch.rand([128, 20], dtype=torch.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([128, 30], dtype=torch.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([40, 20, 30], dtype=torch.float16)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = torch.rand([40], dtype=torch.float16)
arg_4 = arg_4_tensor.clone()
start = time.time()
results["time_low"] = torch.nn.functional.bilinear(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(torch.float32)
arg_2 = arg_2_tensor.clone().type(torch.float32)
arg_3 = arg_3_tensor.clone().type(torch.float32)
arg_4 = arg_4_tensor.clone().type(torch.float32)
start = time.time()
results["time_high"] = torch.nn.functional.bilinear(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
