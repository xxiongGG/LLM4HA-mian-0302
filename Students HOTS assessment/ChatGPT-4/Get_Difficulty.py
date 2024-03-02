import pandas as pd
import torch.nn

data_all_path = "Datasets/20230901/data_all_20230901_1.xlsx"
data_all = pd.read_excel(data_all_path)
data_all = data_all.sort_values(by='Q_id', ascending=False)
data_all['response'] = data_all['response'].astype(str)
q_group = data_all.groupby('Q_id')
difficulty_list = []

Sigmoid = torch.nn.Sigmoid()

for q_id, q_items in q_group:
    response_group = q_items.groupby('response')
    F_count = 0
    C_count = 0
    for response_id, response_items in response_group:
        if response_id == '0':
            F_count = len(response_items)
        if response_id == '1':
            C_count = len(response_items)

    difficulty = F_count / (F_count + C_count)
    difficulty_list.append(round(difficulty, 3))
difficulty_list = [round(i * 10) for i in difficulty_list]
print(difficulty_list)

Q_map_path = "Datasets/20230908/q_list_key.xlsx"
Q_map = pd.read_excel(Q_map_path)
Q_map['Difficulty'] = difficulty_list
Q_map.to_excel("Datasets/20230911/q_list.xlsx", index=False)