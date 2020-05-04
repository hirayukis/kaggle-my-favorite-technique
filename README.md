# kaggle-my-favorite-technique

## overall rule
```python
train = pd.DataFrame()  # ordinally read_csv
test = pd.DataFrame()  # ordinally read_csv

X = train.drop("y", axis=1)
y = train["y"]

cat_features = []  # category
num_features = []
```
