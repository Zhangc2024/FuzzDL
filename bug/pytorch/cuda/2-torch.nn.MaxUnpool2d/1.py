results = dict()
import torch
arg_1 = 43
arg_2 = 2
arg_class = torch.nn.MaxUnpool2d(arg_1,stride=arg_2,)
arg_3_0_tensor = torch.rand([1, 1, 2, 2], dtype=torch.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = torch.randint(-8,8192,[1, 1, 2, 2], dtype=torch.int64)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3_1 = arg_3_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
