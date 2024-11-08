import pandas as pd
df = pd.DataFrame({'id':[1, 3, 5, 10], 'color':['red', 'green', 'pink', 'black']})
df['new_col'] = df.color.apply(lambda x: len(x) - 1)
print(df)