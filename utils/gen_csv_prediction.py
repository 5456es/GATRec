import numpy as np
import pandas as pd
from .compute import cos_sim
from datetime import datetime
import os


def gen_csv_prediction(args,best_embed, refs_to_pred, threshold, test_f1, seed):
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

    time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not os.path.exists(args.predict_path):
        os.makedirs(args.predict_path)

    df.to_csv(f'{args.predict_path}/f1:{test_f1:.5f}-{time}.csv', index=False)
    ### save args as {time}_args.txt
    with open(f'{args.predict_path}/f1:{test_f1:.5f}-{time}_args.txt', 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
