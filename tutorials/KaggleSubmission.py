import pandas as pd
import time

def make_submission_csv(ids, preds, path=None):
    df = pd.DataFrame({'id': ids, 'cancer': preds})
    current_time = int(round(time.time() * 1000))
    if not path:
        path = 'submission_{}.csv'.format(current_time)
    df.to_csv(path, index=False)
    print(df.head())

