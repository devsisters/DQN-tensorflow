import time
import tensorflow as tf

def timeit(f):
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
    return result
  return timed

def inject_summary(writer, tag, value, step):
  summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
  writer.add_summary(summary, global_step=step)

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
