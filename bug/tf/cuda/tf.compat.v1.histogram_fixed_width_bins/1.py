results = dict()
import tensorflow as tf
try:
  try:
    with tf.device('/CPU'):
      values_0 = -1.0
      values_1 = 0.0
      values_2 = 1.5
      values_3 = 2.0
      values_4 = 5.0
      values_5 = 15
      values = [values_0,values_1,values_2,values_3,values_4,values_5,]
      value_range_0 = 0.0
      value_range_1 = False
      value_range = [value_range_0,value_range_1,]
      nbins = 4
      dtype = "tf.dtypes.int32"
      name = None
      results["res_cpu"] = tf.compat.v1.histogram_fixed_width_bins(values=values,value_range=value_range,nbins=nbins,dtype=dtype,name=name,)
  except Exception as e:
    results["err_cpu"] = "Error:"+str(e)
  try:
    with tf.device('/GPU:0'):
      values = [values_0,values_1,values_2,values_3,values_4,values_5,]
      value_range = [value_range_0,value_range_1,]
      results["res_gpu"] = tf.compat.v1.histogram_fixed_width_bins(values=values,value_range=value_range,nbins=nbins,dtype=dtype,name=name,)
  except Exception as e:
    results["err_gpu"] = "Error:"+str(e)
except Exception as e:
  results["err"] = "Error:"+str(e)

print(results)
