results = dict()
import torch
arg_1_tensor = torch.randint(-2048,1,[0, 3, 3], dtype=torch.int16)
arg_1 = arg_1_tensor.clone()
arg_2 = 40
try:
  results["res_cpu"] = torch.cumsum(arg_1,dim=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.cumsum(arg_1,dim=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
