import numpy as np

def computePathloss(gen_para, distances):
    N = np.shape(distances)[-1]
    assert N==gen_para.NofLinks

    h1 = gen_para.tx_height
    h2 = gen_para.rx_height
    signal_lambda = 2.998e8 / gen_para.carrier_f
    antenna_gain_decibel = gen_para.antenna_gain_decibel
    # compute relevant quantity
    Rbp = 4 * h1 * h2 / signal_lambda
    Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * h1 * h2)))
    # compute coefficient matrix for each Tx/Rx pair
    sum_term = 20 * np.log10(distances / Rbp)
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances > Rbp).astype(int)) * sum_term  # adjust for longer path loss
    pathlosses = -Tx_over_Rx + np.eye(N) * antenna_gain_decibel  # only add antenna gain for direct channel
    pathlosses = np.power(10, (pathlosses / 10))  # convert from decibel to absolute
    return pathlosses