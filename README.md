# Data Analysis on Brazilian online store "Olist" 

Data Analysis using the public dataset from the Brazilian online store "Olist" where sellers posting their products and sell.

**Modules to import**

```
import matplotlib.pyplot as plt
import pandas as pd
import os
```

### 1. If they open the offline store, which states of cities should they target first? 


Read each files individually and create dataframes
```
cust_df = pd.read_csv('/Users/kitaeklee/Desktop/Data/brazilian-ecommerce/olist_customers_dataset.csv')
order_df = pd.read_csv('/Users/kitaeklee/Desktop/Data/brazilian-ecommerce/olist_orders_dataset.csv')
seller_df = pd.read_csv('/Users/kitaeklee/Desktop/Data/brazilian-ecommerce/olist_sellers_dataset.csv')
review_df = pd.read_csv('/Users/kitaeklee/Desktop/Data/brazilian-ecommerce/olist_order_reviews_dataset.csv')
productname_df = pd.read_csv('/Users/kitaeklee/Desktop/Data/brazilian-ecommerce/product_category_name_translation.csv')
pay_df = pd.read_csv('/Users/kitaeklee/Desktop/Data/brazilian-ecommerce/olist_order_payments_dataset.csv')
order_item_df = pd.read_csv('/Users/kitaeklee/Desktop/Data/brazilian-ecommerce/olist_order_items_dataset.csv')
location_df = pd.read_csv('/Users/kitaeklee/Desktop/Data/brazilian-ecommerce/olist_geolocation_dataset.csv')
```

Merge the dataframes on the identical variables to match the order_id and customer_id to payment value
```
# merge cust_df and order_df
cust_order_df = pd.merge(order_df, cust_df, on='customer_id', how='inner')

# merge with the pay_df
cust_order_pay_df = pd.merge(cust_order_df, pay_df, on='order_id', how='inner')
```

**Top 5 states with the highest revenues**
```

# create a new dataframe with states and payment_value
state_pay = cust_order_pay_df.loc[:, ['customer_state', 'payment_value']].groupby('customer_state').sum()
state_pay = state_pay.sort_values(by='payment_value', ascending=False).reset_index()
state_pay.head()
```

![Image description](https://i.postimg.cc/VLTQp1XF/Screen-Shot-2020-05-17-at-1-12-47-AM.png)

**Top 5 cities with the highest revenues**
```
# create a new dataframe with cities and total payment
city_pay = cust_order_pay_df.loc[:, ['customer_city', 'payment_value']].groupby('customer_city').sum()
city_pay = city_pay.sort_values(by='payment_value', ascending=False).reset_index()
city_pay.head()
```

![Image description](https://i.postimg.cc/9fFH2PvF/2-city-pay.png)

**Top 50 cities for revenues displayed in bar chart**
```
city_pay_top50 = city_pay.loc[0:49, :]

city_pay_top50.plot(kind='bar', x='customer_city', y='payment_value', figsize=(10,5))
plt.title('Cities with top 50 revenues')
plt.xlabel('Cities of customers')
plt.ylabel('Revenue')
plt.ticklabel_format(axis='y', useOffset=False, style='plain')
plt.show()
```
![Image Description](https://i.postimg.cc/5t649JYG/3-Barchart-for-1.png)

**States with the revenues in descending order displayed in bar chart**
```
state_pay.plot(kind='bar', figsize=(10,5), x='customer_state', y='payment_value')
plt.ticklabel_format(axis='y', style='plain', useOffset=False)
plt.title("States of customers and their revenues")
plt.xlabel("States of customers")
plt.ylabel("Revenue")
```

![Image Description](https://i.postimg.cc/65tzk5F9/4-Barchart-for-1.png)




2. Where does the most revenue coming from?


3. What is the most popular products? 


4. The month with the highest revenues (sales prediction)


5. Customer satisfaction with products(1,2: unsatisfied, 3: moderate, 4,5: satisfied). --> Product Quality 


6. Delivery Performance (Consumption of time till the arrival of the parcels)




6. Who pays more for the delivery fee? (Depends on city and purchase cost)




