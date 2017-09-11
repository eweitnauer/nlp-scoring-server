Development Notes

- the tensorflow backend has problems with loading and using models in a threaded environment
- reading:
    + https://github.com/fchollet/keras/issues/2397
- here are some snippets I unsuccessfuly used to address the issue:
  ```
import threading
import tensorflow as tf
from keras import backend as K
initLock = threading.Lock()
  ...

# in __init__
global initLock
with initLock:
    self.sess = tf.Session()
    K.set_session(self.sess)
    ...
    self.classifier._make_predict_function()
    self.graph = tf.get_default_graph()

# in get_score:
K.set_session(self.sess)
with self.graph.as_default():
    # use self.classifier.predict

```
