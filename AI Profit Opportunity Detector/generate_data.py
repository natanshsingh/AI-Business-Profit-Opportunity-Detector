import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000 


data = pd.DataFrame({
    "transaction_id": range(n),
    "product_id": np.random.randint(1, 50, n),
    "price": np.random.randint(500, 5000, n),
    "discount_percent": np.random.normal(10, 5, n),
    "refund_flag": np.random.choice([0,1], n, p=[0.9,0.1]),
    "salesperson_id": np.random.randint(1, 15, n),
    "region": np.random.choice(["North","South","East","West"], n),
})


data["profit_margin"] = data["price"] * (1 - data["discount_percent"]/100) * np.random.uniform(0.05,0.3,n)


data.loc[data["product_id"].isin([5,7,9]), "price"] *= 0.8

data.loc[data["product_id"].isin([1,2]), "discount_percent"] += 15

data.to_csv("data/transactions.csv", index=False)
print("Synthetic dataset created!")