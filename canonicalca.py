import numpy as np
from sklearn.cross_decomposition import CCA
import reader


def ref_frequencies_for_cca(amount_of_sec, base_frequ, harmonics, amplitude=1):
    """

    :param amount_of_sec:
    :param base_frequ:
    :param harmonics: make sure it includes 1
    :param amplitude:
    :return:
    """
    t = np.linspace(0, amount_of_sec, reader.SAMPLES_PER_SECOND * amount_of_sec, endpoint=True)
    simulated_frequ = []
    for s in harmonics:
        h_sin = amplitude * np.sin((2 * np.pi * base_frequ * s) * t)
        h_cos = amplitude * np.cos((np.pi * 2 * base_frequ * s) * t)
        simulated_frequ.append(h_sin)
        simulated_frequ.append(h_cos)
    return simulated_frequ


def calc_cca(channels, f, amount_of_sec=3):
    '''

    :param channels:
    :param f:
    :param amount_of_sec: of the window
    :return:
    '''
    print('f: ', f, ' amount_s: ', amount_of_sec)
    if f == 7.5:
        p1 = ref_frequencies_for_cca(amount_of_sec, 7.5, [1, 2, 3])
    elif f == 40:
        p1 = ref_frequencies_for_cca(amount_of_sec, 40, [0.5, 1, 2])
    elif f == 12:
        p1 = ref_frequencies_for_cca(amount_of_sec, 12, [1, 2, 3])
    else:
        print('frequency not supported in cca')

    # now windowing:
    window_length = amount_of_sec * reader.SAMPLES_PER_SECOND
    window_shift = 30  # die jungs nutzen wohl 30
    n_comp = 6  # earlier: 6, sklearn default=2

    result_sum = []
    result_max = []
    score_sum = []
    score_max = []

    xs = np.arange(0, channels.shape[0] - window_length, window_shift)
    p1 = np.asarray(p1)

    for i in xs:  # hier soll 1 Zahl pro Zeitschritt rauskommen
        # do cca
        data = channels[i:i + window_length, :]
        cca = CCA(n_components=n_comp)
        cca.fit(data, p1.T)
        X_c, Y_c = cca.transform(data, p1.T)

        result = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=n_comp)
        score = np.diag(np.corrcoef(cca.x_scores_, cca.y_scores_, rowvar=False)[:n_comp, n_comp:])

        result_sum.append((result[0] +result[1] +result[2] +result[3] +result[4] +result[5]))
        result_max.append(np.max(result))
        score_sum.append((score[0] + score[1] + score[2] + score[3] + score[4] + score[5]))
        score_max.append(np.max(score))

    return xs, result_sum, result_max, score_sum, score_max, p1[0]
