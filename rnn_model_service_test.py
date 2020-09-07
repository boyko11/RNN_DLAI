import numpy as np
from rnn_model_service import RNN_ModelService


def rnn_cell_forward_test():

    np.random.seed(1)
    xt_tmp = np.random.randn(3, 10)
    a_prev_tmp = np.random.randn(5, 10)
    parameters_tmp = {}
    parameters_tmp['Waa'] = np.random.randn(5, 5)
    parameters_tmp['Wax'] = np.random.randn(5, 3)
    parameters_tmp['Wya'] = np.random.randn(2, 5)
    parameters_tmp['ba'] = np.random.randn(5, 1)
    parameters_tmp['by'] = np.random.randn(2, 1)

    a_next_tmp, yt_pred_tmp, cache_tmp = RNN_ModelService.rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
    print("a_next[4] = \n", a_next_tmp[4])
    print("a_next.shape = \n", a_next_tmp.shape)
    print("yt_pred[1] =\n", yt_pred_tmp[1])
    print("yt_pred.shape = \n", yt_pred_tmp.shape)


print("rnn_cell_forward_test()")
rnn_cell_forward_test()

def run_forward_test():
    np.random.seed(1)
    x_tmp = np.random.randn(3, 10, 4)
    a0_tmp = np.random.randn(5, 10)
    parameters_tmp = {}
    parameters_tmp['Waa'] = np.random.randn(5, 5)
    parameters_tmp['Wax'] = np.random.randn(5, 3)
    parameters_tmp['Wya'] = np.random.randn(2, 5)
    parameters_tmp['ba'] = np.random.randn(5, 1)
    parameters_tmp['by'] = np.random.randn(2, 1)

    a_tmp, y_pred_tmp, caches_tmp = RNN_ModelService.rnn_forward(x_tmp, a0_tmp, parameters_tmp)
    print("a[4][1] = \n", a_tmp[4][1])
    print("a.shape = \n", a_tmp.shape)
    print("y_pred[1][3] =\n", y_pred_tmp[1][3])
    print("y_pred.shape = \n", y_pred_tmp.shape)
    print("caches[1][1][3] =\n", caches_tmp[1][1][3])
    print("len(caches) = \n", len(caches_tmp))


print("------------------")
print("run_forward_test()")
run_forward_test()

