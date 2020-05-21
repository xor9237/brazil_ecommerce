# Data Analysis on Brazilian online store "Olist" 

Data Analysis using the public dataset from the Brazilian online store "Olist" where sellers posting their products and sell.

**Data Description**

Dataset is consists of 8 datasets. 
- "olist_customers_dataset.csv" consists of customers' ID, customers' zip code, customers' cities and state
- "olist_orders_dataset.csv" consists of order ID, customers' ID, order delivery status, time of purchase, order approved date, order delivered date and order estimated date.
- "olist_sellers_dataset.csv" consists of sellers' ID, sellers' zip codes and sellers' cities and states.
- "olist_order_reviews_dataset.csv" consists of review ID, order ID, review score, review comments and review creation dates.
- "product_category_name_translaton.csv" consists of names or product categories.
- "olist_order_payments_dataset.csv" consists of order ID, payment type, installments of payments and payment values.
- "olist_order_items_dataset.csv" consists of order ID, product ID, seller ID, shipping limit date, price and freight values.
- "olist_geolocation_dataset.csv" consists of zip codes, latitudes, longitudes, cities and states.


**Modules to import**

```
import matplotlib.pyplot as plt
import pandas as pd
import os

import folium
from geopy.geocoders import Nominatim
import matplotlib.cm as cm
import matplotlib.colors as colors
import geocoder

from sklearn.cluster import KMeans

# To get the address by using latitudes and longitudes
import reverse_geocoder as rg
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

### 2. If, for example, Olist first decided to open their first offlines store in Sao Paulo where the highest revenue coming from out of all the cities in Brazil, where would be the place to open the first store?

Locate the location of customers in Sao Paulo, set the radius and choose the location where there are the most customers live around or choose the location where the most social places exist to find the location where it's easily accessible by potential customers. 

The original dataset consist of 1000163 rows which contain the latitudes and longitudes of various cities so I extracted only the rows of the city "Sao Paulo" which has 135800 rows.
```
# Select the Sao Paulo data only
location_saopaulo_df = location_df.loc[location_df['geolocation_city']=='sao paulo'].reset_index()
location_saopaulo_df.drop(columns=['index'], inplace=True)
```
Use reverse_geocoder and latitudes and longitudes from the dataset to get the name of the cities
```
res =  rg.search(tuple(zip(location_saopaulo_df['geolocation_lat'], 
                           location_saopaulo_df['geolocation_lng'])))
df_address = pd.DataFrame(res) 
```




2. What is the most popular products? (Which product is creating revenue the most?)


3. The month with the highest revenues (sales prediction)


4. Customer satisfaction with products(1,2: unsatisfied, 3: moderate, 4,5: satisfied). --> Product Quality 


5. Delivery Performance (Consumption of time till the arrival of the parcels)




6. Who pays more for the delivery fee? (Depends on city and purchase cost)



Elbow method to determine the optimal number of clusters for K-means clustering
