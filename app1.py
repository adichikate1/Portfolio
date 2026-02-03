# y ​= w1​x1 ​+ w2​x2 ​+ ⋯ + wn​xn ​+ b
# w = (X^T*X)^-1*X^T*y

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# ===== MODEL TRAINING (LOAD ONCE) =====
df = pd.read_csv("House-Data.csv")

rem_df = df.drop(
    ["street", "city", "statezip", "country", "waterfront",
     "date", "price", "sqft_above", "sqft_basement"],
    axis=1
)

# Bias term
rem_df.insert(0, "b", 1)

X = rem_df.values
Y = df["price"].values.reshape(-1, 1)

# Normal Equation
W = np.linalg.inv(X.T @ X) @ X.T @ Y
columns = rem_df.columns.tolist()


# ===== ROUTES =====
@app.route("/")
def portfolio():
    return render_template("portfolio.html")   # pehle portfolio

@app.route("/project")
def project():
    return render_template("index.html")       # view project ke baad



@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Order must match CSV columns
    feature_values = [
        1,  # bias
        data["bedrooms"],
        data["bathrooms"],
        data["sqftLiving"],
        data["sqftLot"],
        data["floors"],
        data["balcony"],
        data["rating"],
        data["yearBuilt"],
        data["yearRenovated"]
    ]

    feature_values = np.array(feature_values).reshape(1, -1)

    price = feature_values @ W
    price = int(price[0][0])

    return jsonify({"predicted_price": price})


if __name__ == "__main__":
    app.run()

