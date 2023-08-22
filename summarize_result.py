import os
import pandas as pd
from collections import defaultdict

if __name__ == '__main__':
    result_path='./results'
    interest = 'SimCLR_4096_27_1e-4'
    desired_stages = ['SITTING', '1', '2', '3', '4', 'all', '#1', '#2', '#3', 'resting']
    metric_keys = ['Recall', 'Specificity', 'F1-Score', 'Test Accuracy', 'Test Loss', 'AUROC', 'Test Accuracy (Balanced)', 'Train Loss']
    total_files = [os.path.join(result_path, file) for file in os.listdir(result_path) if file.endswith('.csv')]
    keywords = []
    files = []
    for stage in desired_stages:
        keywords.append('_'.join((interest, stage)))
    for file in total_files:
        for keyword in keywords:
            if keyword in file:
                files.append(file)
    
    final_dict = defaultdict(dict)
    
    for file in files:
        df = pd.read_csv(file)
        stage = file.split('_')[-1].split('.')[0]
        for key in metric_keys:
            if key == 'Recall':
                final_dict[stage]['Sensiticity'] = df[key].to_numpy()[-1]
            elif key == 'Test Accuracy':
                final_dict[stage][key] = df[key].to_numpy()[-1]/100
            else:
                final_dict[stage][key] = df[key].to_numpy()[-1]
    final_df = pd.DataFrame.from_dict(final_dict)
    final_df = final_df[desired_stages]
    final_df.to_csv(f'./result_summaries/{interest}.csv')
    print(f'{interest}.csv SAVED!')