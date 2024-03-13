import torch
from classes.argument import *
from classes.api import *
from classes.database import TorchDatabase
import os
from os.path import join

class TorchArgument(Argument):
    _supported_types = [
        ArgType.TORCH_DTYPE, ArgType.TORCH_OBJECT, ArgType.TORCH_TENSOR
    ]
    _dtypes = [
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        torch.float16, torch.float32, torch.float64, torch.bfloat16,
        torch.complex64, torch.complex128, torch.bool
    ]
    _memory_format = [
        torch.contiguous_format, torch.channels_last, torch.preserve_format
    ]

    def __init__(self,
                 value,
                 type: ArgType,
                 shape=None,
                 dtype=None,
                 max_value=1,
                 min_value=0):
        super().__init__(value, type)
        self.shape = shape
        self.dtype = dtype
        self.max_value = max_value
        self.min_value = min_value

    def to_code(self, var_name, low_precision=False, is_cuda=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}", low_precision,
                                              is_cuda)
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TORCH_TENSOR:
            dtype = self.dtype
            max_value = self.max_value
            min_value = self.min_value
            if low_precision:
                dtype = self.low_precision_dtype(dtype)
                max_value, min_value = self.random_tensor_value(dtype)
            suffix = ""
            if is_cuda:
                suffix = ".cuda()"
            if dtype.is_floating_point:
                code = f"{var_name}_tensor = torch.rand({self.shape}, dtype={dtype})\n"
            elif dtype.is_complex:
                code = f"{var_name}_tensor = torch.rand({self.shape}, dtype={dtype})\n"
            elif dtype == torch.bool:
                code = f"{var_name}_tensor = torch.randint(0,2,{self.shape}, dtype={dtype})\n"
            else:
                code = f"{var_name}_tensor = torch.randint({min_value},{max_value},{self.shape}, dtype={dtype})\n"
            code += f"{var_name} = {var_name}_tensor.clone(){suffix}\n"
            return code
        elif self.type == ArgType.TORCH_OBJECT:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.TORCH_DTYPE:
            return f"{var_name} = {self.value}\n"
        return super().to_code(var_name)

    def to_diff_code(self, var_name, oracle):
        """differential testing with oracle"""
        code = ""
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", oracle)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
        elif self.type == ArgType.TORCH_TENSOR:
            if oracle == OracleType.CUDA:
                code += f"{var_name} = {var_name}_tensor.clone().cuda()\n"
            elif oracle == OracleType.PRECISION:
                code += f"{var_name} = {var_name}_tensor.clone().type({self.dtype})\n"
        return code

    def mutate_value(self) -> None:
        if self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TORCH_TENSOR:
            self.max_value, self.min_value = self.random_tensor_value(
                self.dtype)
        elif self.type in super()._support_types:
            super().mutate_value()
        else:
            print(self.type, self.value)
            assert (0)

    def mutate_type(self) -> None:
        if self.type == ArgType.NULL:        # 如果原来参数类型为空？  代表这个api没有参数？
            # choose from all types
            new_type = choice(self._support_types + super()._support_types)  #随机选一个类型
            self.type = new_type
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:  # 如果选中的类型为 list 或者tuple，还要为里面的随机设值
                self.value = [
                    TorchArgument(2, ArgType.INT),
                    TorchArgument(3, ArgType.INT)
                ]
            elif new_type == ArgType.TORCH_TENSOR:    # 选中tensor
                self.shape = [2, 2]                   # 设置形状
                self.dtype = torch.float32             # 类型
            elif new_type == ArgType.TORCH_DTYPE:
                self.value = choice(self._dtypes)
            elif new_type == ArgType.TORCH_OBJECT:     # 如果new_type 是一个对象类型，随机从_memory_format中挑选一个value
                self.value = choice(self._memory_format)
            else:
                self.value = super().initial_value(new_type)
        elif self.type == ArgType.TORCH_TENSOR:              # 如果参数原类型为 tensor
            new_size = list(self.shape)                      # 获取tensor的形状
            # change the dimension of tensor
            if change_tensor_dimension():
                if add_tensor_dimension():
                    new_size.append(1)
                elif len(new_size) > 0:
                    new_size.pop()
            # change the shape
            for i in range(len(new_size)):
                if change_tensor_shape():
                    new_size[i] = self.mutate_int_value(new_size[i], _min=0)
            self.shape = new_size
            # change dtype
            if change_tensor_dtype():
                self.dtype = choice(self._dtypes)
                self.max_value, self.min_value = self.random_tensor_value(self.dtype)
        elif self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type in super()._support_types:
            super().mutate_type()
        else:
            print(self.type, self.value)
            assert (0)

    # 参数突变组合
    def combined_mutation(self, mutate_info) -> None:
        fields = mutate_info.split("?")
        info_type, info_value = fields[1].split(":")
        if info_type == "ArgType.TORCH_TENSOR" :
            self.type = ArgType.TORCH_TENSOR

            _, dtype_value = fields[3].split(":")
            self.dtype = eval(dtype_value)

            value_type, value_str = fields[2].split(":")
            self.shape = eval(value_type)(value_str)

            value_type, value_str = fields[4].split(":")
            self.max_value = eval(value_type)(value_str)

            value_type, value_str = fields[5].split(":")
            self.min_value = eval(value_type)(value_str)
        elif info_type == "ArgType.LIST" :
            self.type = ArgType.LIST

            self.value = [
                TorchArgument(2, ArgType.INT),
                TorchArgument(3, ArgType.INT)
            ]
        elif info_type == "ArgType.TUPLE" :
            self.type = ArgType.TUPLE

            self.value = [
                TorchArgument(2, ArgType.INT),
                TorchArgument(3, ArgType.INT)
            ]
        elif info_type == "ArgType.TORCH_OBJECT" or info_type == "ArgType.NULL":
            pass
        elif info_type == "ArgType.TORCH_DTYPE":
            self.type = ArgType.TORCH_DTYPE
            _, dtype_value = fields[0].split(":")
            self.value = eval(dtype_value)
        elif info_type == "1" :
            self.type = ArgType.INT

            _, str_value = fields[0].split(":")
            self.value = int(str_value)
        elif info_type == "2" :
            self.type = ArgType.STR

            _, str_value = fields[0].split(":")
            self.value = str_value
        elif info_type == "3" :
            self.type = ArgType.FLOAT

            _, str_value = fields[0].split(":")
            self.value = float(str_value)
        elif info_type == "4" :
            self.type = ArgType.BOOL

            _, str_value = fields[0].split(":")
            self.value = str_value.lower() == "true"
        else:
            pass
            
            
    @staticmethod
    def random_tensor_value(dtype):
        max_value = 1
        min_value = 0
        if dtype == torch.bool:
            max_value = 2
            min_value = 0
        elif dtype == torch.uint8:
            max_value = 1 << randint(0, 9)
            min_value = 0
        elif dtype == torch.int8:
            max_value = 1 << randint(0, 8)
            min_value = -1 << randint(0, 8)
        elif dtype == torch.int16:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        else:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        return max_value, min_value

    @staticmethod
    def generate_arg_from_signature(signature):
        """Generate a Torch argument from the signature"""
        if signature == "torchTensor":    # 参数类型为张量类型 torchTensor
            return TorchArgument(None,
                                 ArgType.TORCH_TENSOR,
                                 shape=[2, 2],
                                 dtype=torch.float32)
        if signature == "torchdtype":
            return TorchArgument(choice(TorchArgument._dtypes),
                                 ArgType.TORCH_DTYPE)
        if isinstance(signature, str) and signature == "torchdevice":      # 变量类型为str类型， 且参数名称为torchdevice
            value = torch.device("cpu")
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torchmemory_format":
            value = choice(TorchArgument._memory_format)
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torch.strided":
            return TorchArgument("torch.strided", ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature.startswith("torch."):
            value = eval(signature)
            if isinstance(value, torch.dtype):
                return TorchArgument(value, ArgType.TORCH_DTYPE)
            elif isinstance(value, torch.memory_format):
                return TorchArgument(value, ArgType.TORCH_OBJECT)
            print(signature)
            assert(0)
        if isinstance(signature, bool):
            return TorchArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):            # 如果参数是 int 型，创建一个TorchArgument对象，并返回
            return TorchArgument(signature, ArgType.INT)
        if isinstance(signature, str):
            return TorchArgument(signature, ArgType.STR)
        if isinstance(signature, float):
            return TorchArgument(signature, ArgType.FLOAT)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.TUPLE)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.LIST)
        # signature is a dictionary
        if isinstance(signature, dict):         # 如果signature是字典类型，则要获取当前参数的 shape 和 dtype
            if not ('shape' in signature.keys()
                    and 'dtype' in signature.keys()):
                raise Exception('Wrong signature {0}'.format(signature))
            shape = signature['shape']
            dtype = signature['dtype']
            # signature is a ndarray or tensor.
            if isinstance(shape, (list, tuple)):
                if not dtype.startswith("torch."):
                    dtype = f"torch.{dtype}"
                dtype = eval(dtype)
                max_value, min_value = TorchArgument.random_tensor_value(dtype)
                return TorchArgument(None,
                                     ArgType.TORCH_TENSOR,
                                     shape,
                                     dtype=dtype,
                                     max_value=max_value,
                                     min_value=min_value)
            else:
                return TorchArgument(None,
                                     ArgType.TORCH_TENSOR,
                                     shape=[2, 2],
                                     dtype=torch.float32)
        return TorchArgument(None, ArgType.NULL)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [torch.int16, torch.int32, torch.int64]:
            return torch.int8
        elif dtype in [torch.float32, torch.float64]:
            return torch.float16
        elif dtype in [torch.complex64, torch.complex128]:
            return torch.complex32
        return dtype

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if isinstance(x, torch.Tensor):
            return ArgType.TORCH_TENSOR
        elif isinstance(x, torch.dtype):
            return ArgType.TORCH_DTYPE
        else:
            return ArgType.TORCH_OBJECT



class TorchAPI(API):
    def __init__(self, api_name, record=None):
        super().__init__(api_name)      # 获取创建对象时的api名称
        if record == None:
            record = TorchDatabase.get_rand_record(self.api)  # 根据创建对象时的api获取记录的 API 或对象。与 API 相关的信息，可能是一些配置或数据。
            # 从数据库中的freefuzz-torch表，根据当前api名称获取一条记录
            # 每条记录形式：input_signature(array[param1:（shape，dtype）, param2:（shape，dtype）]) param1：（shape，dtype）  param2：（shape，dtype） ...   output_signature：（shape，dtype）

        self.args = self.generate_args_from_record(record)
        # 这一行代码将 record 中的信息用于生成参数（args）。这可能是用来构建函数或方法的参数，以便稍后调用 API。
        # 获取当前api有多少个参数？
       # {'parameter:0': < classes.torch_api.TorchArgument object at 0x7f8972cc6d00 >,
       #  'parameter:1': < classes.torch_api.TorchArgument object at 0x7f8972cc6c40 >}

        self.is_class = inspect.isclass(eval(self.api))
        self.file_name = api_name

    def get_param_num(self):        # 获取api中参数数量
        num_arg = len(self.args)
        return num_arg

    def mutate_explore(self, enable_value=True, enable_type=True, enable_db=True, param_index=0):
        params_list = list(self.args.keys())
        arg_name = params_list[param_index]  # 从当前的api的参数列表中随机选取一个参数 arg_name
        arg = self.args[arg_name]  # 每个arge都是一个TorchArgument对象，包括：shape、dtype、max_value、min_value
        if enable_type and do_type_mutation():  # 类型变异   do_type_mutation()  有20%概率返回true 进行类型变异
            arg.mutate_type()
        do_value_mutation = True
        if enable_db and do_select_from_db():
            new_arg, success = TorchDatabase.select_rand_over_db(
                self.api, arg_name)
            if success:
                new_arg = TorchArgument.generate_arg_from_signature(
                    new_arg)
                self.args[arg_name] = new_arg
                do_value_mutation = False
        if enable_value and do_value_mutation:
            arg.mutate_value()
        # 将每次突变的结果以字段的形式写入文件
        fname = f"{self.file_name}.txt"
        file_path = "mutate_log/pytorch/" + fname
        with open(file_path, "a") as file:
            # 定义记录字段
            params_index = str(param_index)
            para_value = str(arg.value)
            para_value_type = type(arg.value)
            para_type = str(arg.type)
            para_type_type = type(arg.type)
            para_shape = str(arg.shape)
            para_shape_type = type(arg.shape)
            para_dtype = str(arg.dtype)
            para_dtype_type = type(arg.dtype)
            para_maxV = str(arg.max_value)
            para_maxV_type = type(arg.max_value)
            para_minV = str(arg.min_value)
            para_minV_type = type(arg.min_value)

            # 将记录字段写入同一行
            field_string = (
                f"{params_index};"
                f"{para_value_type.__name__}:{para_value}?"
                f"{para_type_type.__name__}:{para_type}?"
                f"{para_shape_type.__name__}:{para_shape}?"
                f"{para_dtype_type.__name__}:{para_dtype}?"
                f"{para_maxV_type.__name__}:{para_maxV}?"
                f"{para_minV_type.__name__}:{para_minV};"
            )
            file.write(field_string)

    def mutate_exploit(self, param_num=0, mutate_string = ""):
        parts = mutate_string.split(";")     # 将接收的字符串，按照分号分割，得到param_num部分，每个部分代表对对应参数的突变信息
        for i in range(param_num): # 遍历每个参数的突变信息，并进行相应的突变
            params_list = list(self.args.keys())
            arg_name = params_list[i]  # 从当前的api的参数列表中获取第i个参数 arg_name
            arg = self.args[arg_name]
            mutate_info = parts[i]
            arg.combined_mutation(mutate_info)


    def to_code(self,
                prefix="arg",
                res="res",
                is_cuda=False,
                use_try=False,
                error_res=None,
                low_precision=False) -> str:
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_code(arg_name,
                                low_precision=low_precision,
                                is_cuda=is_cuda)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if is_cuda:
                code += f"{prefix}_class = {self.api}({arg_str}).cuda()\n"
            else:
                code += f"{prefix}_class = {self.api}({arg_str})\n"

            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_code(
                    arg_name, low_precision=low_precision, is_cuda=is_cuda)
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(res, error_res, res_code, use_try,
                                           low_precision)

    def to_diff_code(self,
                     oracle: OracleType,
                     prefix="arg",
                     res="res",
                     *,
                     error_res=None,
                     use_try=False) -> str:
        """Generate code for the oracle"""
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_diff_code(arg_name, oracle)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if oracle == OracleType.CUDA:
                code = f"{prefix}_class = {prefix}_class.cuda()\n"
            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_diff_code(arg_name, oracle)
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(res, error_res, res_code, use_try,
                                           oracle == OracleType.PRECISION)

    @staticmethod
    def invocation_code(res, error_res, res_code, use_try, low_precision):
        code = ""
        if use_try:
            # specified with run_and_check function in relation_tools.py
            if error_res == None:
                error_res = res
            temp_code = "try:\n"
            temp_code += API.indent_code(res_code)
            temp_code += f"except Exception as e:\n  {error_res} = \"ERROR:\"+str(e)\n"
            res_code = temp_code

        if low_precision:
            code += "start = time.time()\n"
            code += res_code
            code += f"{res} = time.time() - start\n"
        else:
            code += res_code
        return code

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:     # 将从数据库获取到的一条api信息
        args = {}                                           # 得到的api信息也是字典的形式？  有多个字段，每个字段下有不同的信息
        # record每条记录形式：“input_signature”：(array[param1:（shape，dtype）, param2:（shape，dtype）]) “param1”：（shape，dtype）  “param2”：（shape，dtype） ...   “output_signature”：（shape，dtype）

        for key in record.keys():
            if key != "output_signature":
                # args[key] 是当前api信息中 key 字段 所对应的参数信息，例如参数的形状、类型等
                args[key] = TorchArgument.generate_arg_from_signature(
                    record[key])  #
        return args
