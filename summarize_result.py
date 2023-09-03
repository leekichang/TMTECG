import os
import pandas as pd
from collections import defaultdict

if __name__ == '__main__':
    result_path='./results'
    interest = 'whole-'
    testsets = ['angio_only_baseline', 'non_angio_baseline', 'whole_baseline']
    metric_keys = ['Sensitivity', 'Specificity', 'F1-Score', 'Test Accuracy', 'Test Loss', 'AUROC', 'Test Accuracy (Balanced)', 'Train Loss']
    total_files = [os.path.join(result_path, file) for file in os.listdir(result_path) if file.endswith('.csv')]
    print(total_files)
    keywords = []
    files = []
    for stage in testsets:
        keywords.append(''.join((interest, stage)))
    for file in total_files:
        for keyword in keywords:
            if keyword in file:
                files.append(file)
    print(files)
    final_dict = defaultdict(dict)
    
    for file in files:
        df = pd.read_csv(file)
        stage = file.split('-')[-1].split('.')[0]
        print(stage)
        for key in metric_keys:
            final_dict[stage][key] = df[key].to_numpy()[-1]
    final_df = pd.DataFrame.from_dict(final_dict)
    final_df = final_df[testsets]
    final_df.to_csv(f'./result_summaries/{interest}.csv')
    print(f'{interest}.csv SAVED!')