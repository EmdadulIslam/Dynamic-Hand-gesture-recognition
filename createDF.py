import pandas as pd
import  numpy as np
from glob import glob


data_list = glob('E:\\ANNmodel\\HGD2full\\*\\*\\*\\*Res3d_feature.npy')
print(data_list)
print(len(data_list))

# featureSample = np.load(data_list[0])

# sample_shape = len(featureSample.tolist()[0])


# df = pd.DataFrame(columns=[f'feature_{i+1}' for i in range(sample_shape)] + ['label', 'split'])


dict_list = []

for i in data_list:

    # print(np.load(i).tolist()[0][0])

    row_data = {f'feature_{i+1}': value for i, value in enumerate(np.load(i).tolist()[0])}
    row_data['label'] = i.split('\\')[-3]
    row_data['split'] = i.split('\\')[-4]

    dict_list.append(row_data)


# print(len(dict_list))

df2 = pd.DataFrame(dict_list)
print(df2.head())

df2.to_csv(".\\Resfeatures_df_HGD2_aug.csv",index=False)
# print(df2)
# print(df2.shape)
# print(df2.columns)

data = pd.read_csv('Resfeatures_df_HGD2_aug.csv')
print("\n")
print(data.shape)
print(data.columns)
#     row_data = np.load(i).tolist()[0]

#     row_data.append(i.split('/')[-3])
#     row_data.append(i.split('/')[-4])

#     df = df.append(row_data, ignore_index=True)


# print(df.head())



