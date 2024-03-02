import openai
import pandas as pd
from tqdm import tqdm


def get_HOT_chatGPT(Q_content):
    openai.api_type = "azure"
    openai.api_base = "your api"
    openai.api_version = "your version"
    openai.api_key = "your key"
    response = openai.ChatCompletion.create(
        engine="gpt4-32k",
        messages=[
            {"role": "system", "content": "You are an expert in the field of education in the subject of C language."},
            {"role": "user",
             "content": "The level of assessment of a topic can be categorized into high and low cognitive levels based on the text of the C topic, in conjunction with Bloom's Cognitive Objective Classification, for example:"
                        "Level 1：Examines understanding and memorization of basic knowledge."
                        "Level 2：Examines the ability to understand and interpret basic knowledge."
                        "Level 3：Examines the ability to apply knowledge in simple ways, writing code and designing functions based on the information in the topic, etc."
                        "Level 4：Examines the ability to analyze and evaluate the structure and effectiveness of programs, including code improvement, bug fixing, and analyzing complex problems."
                        "Level 5：Examines the integrated use of knowledge and skills to creatively solve complex problems. Topics cover several different types of problems and require students to complete more complex and versatile programming and write code to implement them."},
            {"role": "user",
             "content": Q_content + "The above topic C language topic is how to categorize according to Bloom's Cognitive Objectives Taxonomy. Return results in the form of the following"
                                    "[Classification]:[Reason]"}
        ]
    )

    response_str = response['choices'][0]['message']['content']
    return response_str


def get_Q_content(Q_map):
    Q_content_list = Q_map['q_content'].to_list()
    print('【Q_content】 number of Q is :', len(Q_content_list))
    return Q_content_list


def get_Q_HOTs(Q_map):
    HOTs = []
    Q_content_list = get_Q_content(Q_map)
    for Q_content in tqdm(Q_content_list):
        HOT = get_HOT_chatGPT(Q_content)
        print(HOT)
        HOTs.append(HOT)
    Q_map['HOT'] = HOTs
    return Q_map


path = "Datasets/20230908/q_list.xlsx"
Q_map = pd.read_excel(path)
Q_map_1 = Q_map.iloc[:len(Q_map) // 2, :]
Q_map_2 = Q_map.iloc[len(Q_map) // 2:, :]

Q_map_HOTs_1 = get_Q_HOTs(Q_map_1)
Q_map_HOTs_1.to_excel("Datasets/20230908/q_list_hot(1).xlsx", index=False)
print(Q_map_HOTs_1)
Q_map_HOTs_2 = get_Q_HOTs(Q_map_2)
Q_map_HOTs_2.to_excel("CDatasets/20230908/q_list_hot(2).xlsx", index=False)
print(Q_map_HOTs_2)
