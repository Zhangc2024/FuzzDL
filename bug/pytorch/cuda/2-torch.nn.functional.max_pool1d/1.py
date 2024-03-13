results = dict()
import torch
arg_1_tensor = torch.rand([0, 1, 9], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 2
arg_4 = 0
arg_5 = 1
arg_6 = False
arg_7 = True
try:
  results["res_cpu"] = torch.nn.functional.max_pool1d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.nn.functional.max_pool1d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
