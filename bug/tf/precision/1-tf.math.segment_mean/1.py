results = dict()
import tensorflow as tf
import time
try:
  try:
    data_tensor = tf.saturate_cast(tf.random.uniform([2, 3], minval=-256, maxval=257, dtype=tf.int64), dtype=tf.int8)
    data = tf.identity(data_tensor)
    segment_ids_0 = 0
    segment_ids_1 = 1
    segment_ids = [segment_ids_0,segment_ids_1,]
    results["res_low"] = tf.math.segment_mean(data=data,segment_ids=segment_ids,)
    t_start = time.time()
    results["res_low"] = tf.math.segment_mean(data=data,segment_ids=segment_ids,)
    t_end = time.time()
    results["time_low"] = t_end - t_start
  except Exception as e:
    results["err_low"] = "Error:"+str(e)
  try:
    data = tf.identity(data_tensor)
    data = tf.cast(data, tf.int32)
    segment_ids = [segment_ids_0,segment_ids_1,]
    results["res_high"] = tf.math.segment_mean(data=data,segment_ids=segment_ids,)
    t_start = time.time()
    results["res_high"] = tf.math.segment_mean(data=data,segment_ids=segment_ids,)
    t_end = time.time()
    results["time_high"] = t_end - t_start
  except Exception as e:
    results["err_high"] = "Error:"+str(e)
except Exception as e:
  results["err"] = "Error:"+str(e)

print(results)
