results = dict()
import torch
import time
arg_1_tensor = torch.rand([80, 512, 8, 8], dtype=torch.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([1024, 512, 1, 1], dtype=torch.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4_0 = 1
arg_4_1 = 1
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = 0
arg_5_1 = 0
arg_5 = [arg_5_0,arg_5_1,]
arg_6_0 = 1
arg_6_1 = 1
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = 1
start = time.time()
results["time_low"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(torch.float32)
arg_2 = arg_2_tensor.clone().type(torch.float32)
arg_4 = [arg_4_0,arg_4_1,]
arg_5 = [arg_5_0,arg_5_1,]
arg_6 = [arg_6_0,arg_6_1,]
start = time.time()
results["time_high"] = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
results["time_high"] = time.time() - start

print(results)
