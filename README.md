# Data Analysis on Brazilian online store "Olist" 

Data Analysis using the public dataset from the Brazilian online store "Olist" where sellers posting their products and sell.

1. If they open the offline store, which cities should they target first? (Cluster cities into high, mid and low revenues)

Merge the dataframes on the identical variables to match the order_id and customer_id to payment value
```
# merge cust_df and order_df
cust_order_df = pd.merge(order_df, cust_df, on='customer_id', how='inner')

# merge with the pay_df
cust_order_pay_df = pd.merge(cust_order_df, pay_df, on='order_id', how='inner')
```

![GitHub Logo](/Users/kitaeklee/Desktop/Data/brazilian-ecommerce/images/1.state_pay.png)
Format: ![Alt Text](https://postimg.cc/56wRQh7H)



2. Where does the most revenue coming from?


3. What is the most popular products? 


4. The month with the highest revenues (sales prediction)


5. Customer satisfaction with products(1,2: unsatisfied, 3: moderate, 4,5: satisfied). --> Product Quality 


6. Delivery Performance (Consumption of time till the arrival of the parcels)




6. Who pays more for the delivery fee? (Depends on city and purchase cost)
