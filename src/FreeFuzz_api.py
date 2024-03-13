import sys
from constants.enum import OracleType
import configparser
from os.path import join
from utils.converter import str_to_bool
import time
import random

if __name__ == "__main__":
    config_name = sys.argv[1]          # argv = [config_name, "torch", api_name]
    library = sys.argv[2]
    api_name = sys.argv[3]

    freefuzz_cfg = configparser.ConfigParser()
    freefuzz_cfg.read(join(__file__.replace("FreeFuzz_api.py", "config"), config_name))

    # database configuration
    mongo_cfg = freefuzz_cfg["mongodb"]
    host = mongo_cfg["host"]
    port = int(mongo_cfg["port"])

    # oracle configuration
    oracle_cfg = freefuzz_cfg["oracle"]
    crash_oracle = str_to_bool(oracle_cfg["enable_crash"])
    cuda_oracle = str_to_bool(oracle_cfg["enable_cuda"])
    precision_oracle = str_to_bool(oracle_cfg["enable_precision"])

    diff_bound = float(oracle_cfg["float_difference_bound"])
    time_bound = float(oracle_cfg["max_time_bound"])
    time_thresold = float(oracle_cfg["time_thresold"])

    # output configuration
    output_cfg = freefuzz_cfg["output"]
    torch_output_dir = output_cfg["torch_output"]
    tf_output_dir = output_cfg["tf_output"]

    # mutation configuration
    mutation_cfg = freefuzz_cfg["mutation"]         # 查询配置文件中 mutation 字段的内容mutation_cfg
    enable_value = str_to_bool(mutation_cfg["enable_value_mutation"])
    enable_type = str_to_bool(mutation_cfg["enable_type_mutation"])
    enable_db = str_to_bool(mutation_cfg["enable_db_mutation"])
    each_api_run_times = int(mutation_cfg["each_api_run_times"])
    # 查询获取的mutation_cfg中 each_api_run_times字段的值
    # 每个api执行的次数

    if library.lower() in ["pytorch", "torch"]:     # 如果运行指令中，指明要测试的库为 pytorch 或者 torch
        import torch
        from classes.torch_library import TorchLibrary
        from classes.torch_api import TorchAPI
        from classes.database import TorchDatabase
        from utils.skip import need_skip_torch

        TorchDatabase.database_config(host, port, mongo_cfg["torch_database"])   # 连接数据库中的 freefuzz-torch

        if cuda_oracle and not torch.cuda.is_available():
            print("YOUR LOCAL DOES NOT SUPPORT CUDA")
            cuda_oracle = False
        # Pytorch TEST
        MyTorch = TorchLibrary(torch_output_dir, diff_bound, time_bound,    # 实例化一个对象MyTorch， 该对象包含TorchLibrary中的所有方法
                            time_thresold)
        for _ in range(each_api_run_times):  # 执行each_api_run_times次的循环
            api = TorchAPI(api_name)              # 使用当前遍历到的api名称api_name 声明一个TorchAPI的对象 api，包含类TorchAPI其中的所有方法
            print(api_name)
            # ---------------------------------------------------
            mutate_num = 50 # 对每个参数的突变次数
            params_num = api.get_param_num()
            print("---- start explore ---- api name: "+ api_name + " ----")
            print("---- explore total mutate nums:" + str(mutate_num * params_num))
            for i in range(params_num):
                for j in range(mutate_num):           # mutate_num： 对参数i的突变次数
                    api.mutate_explore(enable_value, enable_type, enable_db, i)
                    # 突变完成，运行，记录运行时间
                    start_time = time.time()
                    if crash_oracle:
                        MyTorch.test_with_oracle(api, OracleType.CRASH)
                    if cuda_oracle:
                        MyTorch.test_with_oracle(api, OracleType.CUDA)
                    if precision_oracle:
                        MyTorch.test_with_oracle(api, OracleType.PRECISION)
                    end_time = time.time()
                    execution_time = end_time - start_time

                    # 将执行信息写入文件记录
                    fname = f"{api_name}.txt"
                    file_path = "mutate_log/pytorch/" + fname
                    with open(file_path, "a") as file:
                        # 定义记录字段
                        exe_time = str(execution_time)

                        # 将记录字段写入同一行
                        file.write(f"{exe_time}\n")
            # ---------------------------------------------------
            # 在探索阶段，对api的每个参数都变异mutate_num次，接下来就是选取最优的突变组合
            # 创建字典，用于存储记录和执行时间
            record_dict = {}

            # 打开文件以读取记录
            fname = f"{api_name}.txt"
            file_path = "mutate_log/pytorch/" + fname
            with open(file_path, "r") as file:
                for line in file:
                    # 分割每行以获取第一部分、第二部分、第三部分；分别存放的是：参数索引、参数突变信息、突变执行时间
                    parts = line.strip().split(";")

                    if len(parts) == 3:
                        index, string, exe_time = parts

                        # 将信息添加到相应的 "index" 键对应的列表中
                        if index in record_dict:
                            record_dict[index].append((string, exe_time))
                        else:
                            record_dict[index] = [(string, exe_time)]

            # 对每个列表中的记录按照 "exe_time" 字段从大到小排序
            for index, records in record_dict.items():
                records.sort(key=lambda x: float(x[1]), reverse=True)

            # 创建字典，用于存储每个 "index" 参数突变中，执行时间前m大的突变情况 对应的列表（有多少个参数，就有多少个列表）
            result_dict = {}

            # 假设取出每个列表的前 10 项（例如 m=10），这10项代表了对当前参数的n次突变中，执行时间最长的10次突变信息
            m = 10

            num_p = 0   # api中参数的个数
            # 遍历每个列表，然后使用切片来获取前 m 项
            for index, records in record_dict.items():
                result_dict[index] = []  # 创建一个空列表用于存放结果
                num_p += 1
                for record in records[:m]:
                    # 创建一个新记录，去掉 "exe_time" 字段
                    record_without_exe_time = (record[0])  # 使用切片去掉 "exe_time" 字段
                    result_dict[index].append(record_without_exe_time)


            # ---------------------------------------------------
            # 下面是利用阶段
            print("---- enter exploit ---- api name: " + api_name + " ----")
            exploit_num = num_p * m        # 利用阶段的突变次数
            print("---- exploit total mutate nums:" + str(exploit_num))
            for i in range(exploit_num):
                # 遍历 result_dict 中的每个列表
                result_strings = []  # 用于存放结果字符串
                num_params = 0
                for index, records in result_dict.items():
                    # 从当前列表中随机选择一项
                    random_record = random.choice(list(records))
                    # print(random_record)

                    # 添加到结果字符串列表中
                    result_strings.append(random_record)
                    num_params += 1
                # 将结果字符串列表拼接成一个以分号分隔的字符串
                # print(result_strings)
                result_string = ";".join(result_strings)
                api.mutate_exploit(num_params, result_string)
                if crash_oracle:
                    MyTorch.test_with_oracle(api, OracleType.CRASH)
                if cuda_oracle:
                    MyTorch.test_with_oracle(api, OracleType.CUDA)
                if precision_oracle:
                    MyTorch.test_with_oracle(api, OracleType.PRECISION)
    elif library.lower() in ["tensorflow", "tf"]:
        import tensorflow as tf
        from classes.tf_library import TFLibrary
        from classes.tf_api import TFAPI
        from classes.database import TFDatabase
        from utils.skip import need_skip_tf

        TFDatabase.database_config(host, port, mongo_cfg["tf_database"])
        if cuda_oracle and not tf.test.is_gpu_available():
            print("YOUR LOCAL DOES NOT SUPPORT CUDA")
            cuda_oracle = False
        
        MyTF = TFLibrary(tf_output_dir, diff_bound, time_bound,
                            time_thresold)
        print(api_name)
        if need_skip_tf(api_name): pass
        else:
            for _ in range(each_api_run_times):
                api = TFAPI(api_name)
                print(api_name)
                # ---------------------------------------------------
                mutate_num = 100  # 对每个参数的突变次数
                params_num = api.get_param_num()
                print("---- start explore ---- api name: " + api_name + " ----")
                print("---- explore total mutate nums:" + str(mutate_num * params_num))
                for i in range(params_num):
                    for j in range(mutate_num):  # mutate_num： 对参数i的突变次数
                        api.mutate_explore(enable_value, enable_type, enable_db, i)
                        # 突变完成，运行，记录运行时间
                        start_time = time.time()
                        if crash_oracle:
                            MyTF.test_with_oracle(api, OracleType.CRASH)
                        if cuda_oracle:
                            MyTF.test_with_oracle(api, OracleType.CUDA)
                        if precision_oracle:
                            MyTF.test_with_oracle(api, OracleType.PRECISION)
                        end_time = time.time()
                        execution_time = end_time - start_time

                        # 将执行信息写入文件记录
                        fname = f"{api_name}.txt"
                        file_path = "mutate_log/tf/" + fname
                        with open(file_path, "a") as file:
                            # 定义记录字段
                            exe_time = str(execution_time)

                            # 将记录字段写入同一行
                            file.write(f"{exe_time}\n")
                # ---------------------------------------------------
                # 在探索阶段，对api的每个参数都变异mutate_num次，接下来就是选取最优的突变组合
                # 创建字典，用于存储记录和执行时间
                record_dict = {}

                # 打开文件以读取记录
                fname = f"{api_name}.txt"
                file_path = "mutate_log/tf/" + fname
                with open(file_path, "r") as file:
                    for line in file:
                        # 分割每行以获取第一部分、第二部分、第三部分；分别存放的是：参数索引、参数突变信息、突变执行时间
                        parts = line.strip().split(";")

                        if len(parts) == 3:
                            index, string, exe_time = parts

                            # 将信息添加到相应的 "index" 键对应的列表中
                            if index in record_dict:
                                record_dict[index].append((string, exe_time))
                            else:
                                record_dict[index] = [(string, exe_time)]

                # 对每个列表中的记录按照 "exe_time" 字段从大到小排序
                for index, records in record_dict.items():
                    records.sort(key=lambda x: float(x[1]), reverse=True)

                # 创建字典，用于存储每个 "index" 参数突变中，执行时间前m大的突变情况 对应的列表（有多少个参数，就有多少个列表）
                result_dict = {}
                # 假设取出每个列表的前 10 项（例如 m=10），这10项代表了对当前参数的n次突变中，执行时间最长的10次突变信息
                m = 10 

                num_p = 0  # api中参数的个数
                # 遍历每个列表，然后使用切片来获取前 m 项
                for index, records in record_dict.items():
                    result_dict[index] = []  # 创建一个空列表用于存放结果
                    num_p += 1
                    for record in records[:m]:
                        # 创建一个新记录，去掉 "exe_time" 字段
                        record_without_exe_time = (record[0])  # 使用切片去掉 "exe_time" 字段
                        result_dict[index].append(record_without_exe_time)


                # ---------------------------------------------------
                # 下面是利用阶段
                print("---- enter exploit ---- api name: " + api_name + " ----")
                #exploit_num = num_p * m  # 利用阶段的突变次数
                exploit_num = 30
                print("---- exploit total mutate nums:" + str(exploit_num))
                for i in range(exploit_num):
                    # 遍历 result_dict 中的每个列表
                    result_strings = []  # 用于存放结果字符串
                    num_params = 0
                    for index, records in result_dict.items():
                        # 从当前列表中随机选择一项
                        random_record = random.choice(list(records))
                        # print(random_record)

                        # 添加到结果字符串列表中
                        result_strings.append(random_record)
                        num_params += 1
                    # 将结果字符串列表拼接成一个以分号分隔的字符串
                    # print(result_strings)
                    result_string = ";".join(result_strings)
                    api.mutate_exploit(num_params, result_string)
                    if crash_oracle:
                        MyTF.test_with_oracle(api, OracleType.CRASH)
                    if cuda_oracle:
                        MyTF.test_with_oracle(api, OracleType.CUDA)
                    if precision_oracle:
                        MyTF.test_with_oracle(api, OracleType.PRECISION)
    else:
        print(f"WE DO NOT SUPPORT SUCH DL LIBRARY: {library}!")

