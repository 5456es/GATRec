import numpy as np
import pandas as pd
from .compute import cos_sim
import os
def gen_csv_prediction(best_embed, refs_to_pred, threshold, test_f1, seed):
    # Output your prediction
    test_arr = np.array(refs_to_pred)
    res = cos_sim(np.array(best_embed['author'][test_arr[:, 0]]), np.array(best_embed['paper'][test_arr[:, 1]]))
    res[res >= threshold] = 1
    res[res < threshold] = 0
    data = []
    for index, p in enumerate(list(res)):
        tp = [index, str(int(p))]
        data.append(tp)

    if not os.path.exists('new_prediction'):
        os.makedirs('new_prediction')
    df = pd.DataFrame(data, columns=["Index", "Predicted"], dtype=object)
    df.to_csv(f'new_prediction/NEW_PREDICTION~{test_f1:.5f}-{seed}.csv', index=False)