# Data Analysis on Brazilian online store "Olist" 

Data Analysis using the public dataset from the Brazilian online store "Olist" where sellers posting their products and sell.

### 1. If they open the offline store, which states of cities should they target first? 

Merge the dataframes on the identical variables to match the order_id and customer_id to payment value
```
# merge cust_df and order_df
cust_order_df = pd.merge(order_df, cust_df, on='customer_id', how='inner')

# merge with the pay_df
cust_order_pay_df = pd.merge(cust_order_df, pay_df, on='order_id', how='inner')
```

### Top 5 states with the highest revenues
```
# create a new dataframe with states and payment_value
state_pay = cust_order_pay_df.loc[:, ['customer_state', 'payment_value']].groupby('customer_state').sum()
state_pay = state_pay.sort_values(by='payment_value', ascending=False).reset_index()
state_pay.head()
```
![Image description](https://i.postimg.cc/VLTQp1XF/Screen-Shot-2020-05-17-at-1-12-47-AM.png)

### Top 5 cities with the highest revenues
```
# create a new dataframe with cities and total payment
city_pay = cust_order_pay_df.loc[:, ['customer_city', 'payment_value']].groupby('customer_city').sum()
city_pay = city_pay.sort_values(by='payment_value', ascending=False).reset_index()
city_pay.head()
```
![Image description](https://i.postimg.cc/9fFH2PvF/2-city-pay.png)



2. Where does the most revenue coming from?


3. What is the most popular products? 


4. The month with the highest revenues (sales prediction)


5. Customer satisfaction with products(1,2: unsatisfied, 3: moderate, 4,5: satisfied). --> Product Quality 


6. Delivery Performance (Consumption of time till the arrival of the parcels)




6. Who pays more for the delivery fee? (Depends on city and purchase cost)




