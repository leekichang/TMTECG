import os
import traceback
import pandas as pd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
"""
code reference
https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
"""
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

if __name__ == '__main__':
    save_path = './results'
    os.makedirs(save_path, exist_ok=True)
    tb_path = './tensorboard'
    interest = 'whole-'
    folders = [os.path.join(tb_path, folder) for folder in os.listdir(tb_path) if interest in folder]
    metric_keys = ['Sensitivity', 'Specificity', 'F1-Score', 'Test Accuracy', 'Test Loss', 'AUROC', 'Test Accuracy (Balanced)', 'Train Loss']
    metrics_dict = defaultdict(list)
    for folder in folders:
        file = os.listdir(folder)
        df = tflog2pandas(f'{folder}/{file[0]}')
        for k in metric_keys:
            metrics_dict[k] = df.loc[df['metric']==k]['value']
        df = pd.DataFrame.from_dict(metrics_dict)
        file_name = result = '_'.join(folder.split('_')[2:])
        df.to_csv(f"{save_path}/{file_name}.csv")
        print(f"{save_path}/{file_name}.csv SAVED!")