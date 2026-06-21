import joblib
import pandas as pd

model = joblib.load("car_price_model.pkl")

name = input("Car Name: ")
company = input("Company: ")
year = int(input("Year: "))
kms = int(input("Kilometers Driven: "))
fuel = input("Fuel Type (Petrol/Diesel): ")

new_car = pd.DataFrame({
    "name": [name],
    "company": [company],
    "year": [year],
    "kms_driven": [kms],
    "fuel_type": [fuel]
})

price = model.predict(new_car)

print(f"\nEstimated Price: ₹{price[0]:,.0f}")