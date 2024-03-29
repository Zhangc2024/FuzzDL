results = dict()
import torch
arg_1_tensor = torch.rand([0, 0], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 62
try:
  results["res_cpu"] = torch.cumprod(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.cumprod(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
