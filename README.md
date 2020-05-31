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

### 1. Where are the cities and the states with the most revenues? 


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

### 2. If, for example, Olist first decided to open their first offline store for clothes in Sao Paulo where the highest revenue coming from out of all the cities in Brazil, which area should Olist consider?

**Plan: Locate the location of customers in Sao Paulo, set the radius and choose the location where there are the most customers live around or choose the location where the most social places exist to find the location where it's easily accessible by potential customers.**

- The original dataset consist of 1000163 rows which contain the latitudes and longitudes of various cities so I extracted only the rows of the city "Sao Paulo" which has 135800 rows.
```
# Select the Sao Paulo data only
location_saopaulo_df = location_df.loc[location_df['geolocation_city']=='sao paulo'].reset_index()
location_saopaulo_df.drop(columns=['index'], inplace=True)
```

- Use reverse_geocoder and latitudes and longitudes from the dataset to get the name of the cities
```
res =  rg.search(tuple(zip(location_saopaulo_df['geolocation_lat'], 
                           location_saopaulo_df['geolocation_lng'])))
df_address = pd.DataFrame(res) 
```

- Create a map of Sao Paulo
```
location_saopaulo = geolocator.geocode("Sao Paulo, Brazil")
map_saopaulo = folium.Map(location=[location_saopaulo.latitude, location_saopaulo.longitude])
```
![Image Description](https://i.postimg.cc/vmSvsDjD/5-saopaulo-map.png)

- Cluster the given data of latitudes and longitudes into the cities in Sao Paulo
```
from folium.plugins import FastMarkerCluster
             
callback = ('function (row) {' 
                'var marker = L.marker(new L.LatLng(row[0], row[1]), {color: "red"});'
                'var icon = L.AwesomeMarkers.icon({'
                "icon: 'info-sign',"
                "iconColor: 'white',"
                "markerColor: 'green',"
                "prefix: 'glyphicon',"
                "extraClasses: 'fa-rotate-0'"
                    '});'
                'marker.setIcon(icon);'
                "var popup = L.popup({maxWidth: '300'});"
                "const display_text = {text: row[2]};"
                "var mytext = $(`<div id='mytext' class='display_text' style='width: 100.0%; height: 100.0%;'> ${display_text.text}</div>`)[0];"
                "popup.setContent(mytext);"
                "marker.bindPopup(popup);"
                'return marker};')
                             
map_saopaulo.add_child(FastMarkerCluster(df_address[['lat', 'lon','name']].values.tolist(), callback=callback))
```
![Image Description](https://i.postimg.cc/tCw2cfB8/6-saopaulo-clusteredmap.png)
Among the cities in Sao Paulo, Brazil, the most accessible city and where the most Olist's customers are living in is city Sao Paulo in the state Sao Paulo.

- Find the best Borough to open the clothing store

Since now I know that the best city is Sao Paulo in the state Sao Paulo, I found the list of boroughs in Sao Paulo and made the list then create the new dataframe containing the list of Boroughs.
```
# Which borough shoud Olist choose to open the store in Sao Paulo
# Get the boroughs in Sao Paulo and get the latitude and longitude of each
borough_saopaulo = pd.DataFrame(columns=['borough'])
list_borough = ['Aricanduva', 'Butantã', 'Campo Limpo', 'Casa Verde',
       'Cidade Ademar', 'Cidade Tiradentes', 'Ermelino Matarazzo', 'Freguesia-Brasilândia',
       'Guaianases', 'Ipiranga', 'Itaim Paulista', 'Itaquera', 'Jabaquara', 'Jaçanã-Tremembé',
       'Lapa', 'M\'Boi Mirim', 'Mooca', 'Parelheiros', 'Penha', 'Perus', 'Pinheiros',
       'Pirituba-Jaraguá', 'Santana-Tucuruvi', 'Santo Amaro', 'São Mateus', 'São Miguel Paulista',
       'Sapopemba', 'Sé', 'Villa Guilherme', 'Vila Mariana',
       'Vila Prudente']
num=0
for x in list_borough:
    borough_saopaulo.loc[num, 'borough'] = x
    num+=1
```

Add the empty columns of latitude and longitude to the new dataframe. Then, use the name of boroughs to get latitudes and longitudes of each borough.
```
# Create empty colums for latitude and longitude
borough_saopaulo['latitude']=""
borough_saopaulo['longitude']=""

# Use name of Boroughs to get latitude and logitude and apply to the new columns
y=0
for x in borough_saopaulo.loc[:,'borough']:
    geolocator = Nominatim(user_agent="foursquare_agent")
    location = geolocator.geocode("{}, São Paulo, Brazil".format(x))
    borough_saopaulo.loc[y, 'latitude'] = location.latitude
    borough_saopaulo.loc[y, 'longitude'] = location.longitude
    y+=1
```
First 5 rows of the new dataframe which consist of the name of borough, latitude and longitude as an exmaple.
![Image Description](https://i.postimg.cc/QNzLNHHK/7-df-borough-saopaulo.png)

- Define the Foursquare information and radius
```
# Define Foursquare Credentials and Version
CLIENT_ID = '3MAR2Y4AH5PUQKHSI4TCMUQMLU2S45MO0TVJAKMHDDHAGINK' #  Foursquare ID
CLIENT_SECRET = 'SRPNGW0J3R5EFCSNVENLIXE4BUCCI0X2EEDBYRZTGYLY4HL3' #  Foursquare Secret
VERSION = '20180605' # Foursquare API version

# Define the radius 
radius=3000    # radius in meters
LIMIT = 200    # limit the number of venues to return
```

- Define the function for getting the venues
```
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Borough', 
                  'Borough Latitude', 
                  'Borough Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```

- Use the defined function to get the venues
```
# def function on each neighborhood and create a new dataframe
saopaulo_venues = getNearbyVenues(names=borough_saopaulo['borough'],
                                   latitudes=borough_saopaulo['latitude'],
                                   longitudes=borough_saopaulo['longitude']
                                  )
```

- Check the list of venues to see if there are venues to replace for better clustering result
```
# List of all venues (duplicates removed)
all_venues_list = list(saopaulo_venues['Venue Category'])
venues_list=[]
for venue in all_venues_list:
    if venue not in venues_list:
        venues_list.append(venue)
venues_list
```
Example of the venues
![Image Description](https://i.postimg.cc/x19gR7dN/8-list-of-venues.png)

- **Data Preprocessing**  :  Since venues in 'Venue Category' have duplicates but with different names, replace the values if they can be categorized into same name of the venue.
For example, Sake Bar and Lounge will be replaced to 'Bar'
List of Venues
```
# List of venues to replace the values
list_gym = ['Gym / Fitness Center', 'Gymnastics Gym', 'College Gym', 'Gym']
list_leisure=['Pool', 'Gym Pool', 'Spa', 'Skate Park', 'Bowling Alley', 
              'Pool Hall', 'Outdoors & Recreation', 'Arcade']
list_theater = ['Theater', 'Movie Theater', 'Multiplex', 'Indie Theater']
list_park = ['Skate Park']
list_sports = ['Athletics & Sports','Martial Arts Dojo', 'Dance Studio']
list_station = ['Metro Station', 'Train Station', 'Bus Station']
list_art = ['Art Gallery', 'Art Museum', 'Science Museum', 'History Museum',
           'Street Art', 'Museum']
list_pastry = ['Pastry Shop', 'Bakery', 'Pastelaria', 'Pie Shop']
list_drink = ['Bar', 'Sake Bar', 'Lounge', 'Brewery', 'Beer Store',
             'Beer Bar', 'Gastropub', 'Speakeasy', 'Dive Bar', 
             'Beer Garden']
list_dessert = ['Creperie', 'Cupcake Shop', 'Candy Store', 'Dessert Shop',
               'Ice Cream Shop', 'Snack Place', 'Chocolate Shop',
               'Acai House']
list_fashion = ['Men\'s Store', 'Women\'s Store', 'Thrift / Vintage Store',
               'Shoe Store', 'Jewelry Store', 'Costume Shop', 'Outlet Store']
list_restaurant = ['Vegetarian / Vegan Restaurant', 'Asian Restaurant',
                  'Brazilian Restaurant', 'Diner', 'Mineiro Restaurant',
                  'Fried Chicken Joint', 'Steakhouse', 'Burger Joint',
                  'Pizza Place', 'Food', 'Wings Joint', 'BBQ Joint',
                  'Comfort Food Restaurant', 'Food & Drink Shop',
                  'Food Stand', 'Buffet', 'Deli / Bodega', 
                  'Italian Restaurant', 'American Restaurant',
                  'Paella Restaurant', 'Seafood Restaurant', 'Bistro',
                  'Sandwich Place', 'Japanese Restaurant', 'Argentinian Restaurant',
                  'Sushi Restaurant', 'Food Truck', 'Breakfast Spot',
                  'Persian Restaurant', 'Empanada Restaurant', 'Bagel Shop',
                  'Hot Dog Joint', 'Fast Food Restaurant', 'Cafeteria',
                  'Mexican Restaurant', 'Middle Eastern Restaurant', 
                  'Chinese Restaurant', 'Falafel Restaurant', 'Northern Brazilian Restaurant',
                  'Tapiocaria', 'Noodle House', 'Halal Restaurant', 'Empada House',
                  'Churrascaria']
list_artvenue = ['Performing Arts Venue', 'Music Venue', 'Arts & Entertainment',
                'Concert Hall', 'Auditorium']
list_grocery = ['Grocery Store', 'Fruit & Vegetable Store', 'Supermarket'
               ,'Farmers Market', 'Market', 'Convenience Store']
list_home = ['Mattress Store', 'Furniture / Home Store']
list_mall = ['Plaza', 'Department Store', 'Mall', 'Pedestrian Plaza']
list_cafe = ['Tea Room', 'Juice Bar', 'Coffee Shop', 'Café', 'Cafe']
list_club = ['Rock Club', 'Nightclub', 'Club']
list_accomodations = ['Hotel', 'Motel', 'Hostel', 'Bed & Breakfast']
list_pharmacy = ['Drugstore', 'Pharmacy']

# Replace the values of venues
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_gym, 'Gym')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_leisure, 'Leisure & Recreation')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_theater, 'Theater')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_park, 'Park')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_sports, 'Sports Studio')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_station, 'Public Transportation')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_art, 'Gallery & Museum')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_pastry, 'Pastry & Bakery')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_drink, 'Pub & Bar')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_dessert, 'Dessert')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_fashion, 'Fashion Store')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_restaurant, 'Restaurant')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_artvenue, 'Art Venues')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_grocery, 'Grocery')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_home, 'Home Store')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_mall, 'Mall')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_cafe, 'Cafe')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_club, 'Club')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_accomodations, 'Accomodations')
saopaulo_venues['Venue Category'] = saopaulo_venues['Venue Category'].replace(list_pharmacy, 'Pharmacy')
```

- **"One hot encoding"** to convert categorical variables to numeric variables.
```
saopaulo_onehot = pd.get_dummies(saopaulo_venues[['Venue Category']], prefix="", prefix_sep="")

# add borough column back to dataframe
saopaulo_onehot['Borough'] = saopaulo_venues['Borough'] 

# move borough column to the first column
saopaulo_onehot = saopaulo_onehot.set_index('Borough').reset_index()
```
- Group rows by borough and by taking the mean of the frequency of occurence of each category then write a function that sort the venues in descending order.
```
saopaulo_grouped = saopaulo_onehot.groupby('Borough').mean().reset_index()

# Write a function to sort the venues in descending order.
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```
- Create a new dataframe and display top 10 venues for each borough.
```
# Create a new dataframe and display top 10 venues for each borough
import numpy as np
num_top_venues = 5

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Borough']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
borough_venues_sorted = pd.DataFrame(columns=columns)
borough_venues_sorted['Borough'] = saopaulo_grouped['Borough']

for ind in np.arange(saopaulo_grouped.shape[0]):
    borough_venues_sorted.iloc[ind, 1:] = return_most_common_venues(saopaulo_grouped.iloc[ind, :], num_top_venues)
```
- **Find the optimal 'K' for K-means clustering.**
- Drop the column 'Borough' for clustering
```
saopaulo_grouped_clustering = saopaulo_grouped.drop('Borough', 1)
```

- **Elbow Method**
```
# Use Elbow Method to find the optimal 'k' (number of clusters)
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(saopaulo_grouped_clustering)
    distortions.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```
![Image Description](https://i.postimg.cc/6QrqQkzQ/9-Elbow-Method.png)
However, Elbow Method does not show "elbow" where it shows the optimal "k". Therefore, I used the "Silhouette Method" to find the optimal number of clusters

- **Silhouette Method**
```
from sklearn.metrics import silhouette_score

sil=[ ]
kmax=10
K=range(2,kmax+1)

#dissimilarity would not be defined for a single cluster, thus minimum number of clusters should be 2
for k in K:
    kmeans = KMeans(n_clusters = k).fit(saopaulo_grouped_clustering)
    labels = kmeans.labels_
    sil.append(silhouette_score(saopaulo_grouped_clustering, labels, metric = 'euclidean'))
    
plt.plot(K, sil)
plt.xlabel('k')
plt.ylabel('Silhoutte method')
```
![Image Description](https://i.postimg.cc/66j3TLVC/10-Silhouette-Method.png)
Based on the result of the Silhouette Method, the optimal number of clusters is "7".

- Run K-Means Clustering with k=7 and create a new dataframe with cluster labels.
```
# Based on the Silhouette Method, the optimal number of clusters is 7
kclusters=7

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(saopaulo_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 

# Create a new dataframe that includes the cluster as well as the top 10 venues for each borough.
# add clustering labels

borough_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_) 

saopaulo_merged = borough_saopaulo

# merge saopaulo_grouped with saopaulo_data to add latitude/longitude for each borough
saopaulo_merged = saopaulo_merged.join(borough_venues_sorted.set_index('Borough'), on='borough')
```
Example of the new dataframe
![Image Description](https://i.postimg.cc/6Qx1spcT/11-saopaulo-merged.png)

- Visualization of the map of K-means clustering
```
# Visualize
# create map
location_saopaulo = geolocator.geocode("Sao Paulo, Sao Paulo, Brazil")
map_clusters = folium.Map(location=[location_saopaulo.latitude, location_saopaulo.longitude], zoom_start=10)


# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(saopaulo_merged['latitude'], saopaulo_merged['longitude'], saopaulo_merged['borough'], saopaulo_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
```
![Image Description](https://i.postimg.cc/CKJCvh63/12-Map-clustered.png)


- To examine the clusters, return the dataframe of each cluster with top 5 venues
```
# Examine cluster 1
saopaulo_merged.loc[saopaulo_merged['Cluster Labels'] == 0, 
                     saopaulo_merged.columns[[0] + list(range(4, saopaulo_merged.shape[1]))]]
# Examine cluster 2
saopaulo_merged.loc[saopaulo_merged['Cluster Labels'] == 1, 
                     saopaulo_merged.columns[[0] + list(range(4, saopaulo_merged.shape[1]))]]
# Examine cluster 3
saopaulo_merged.loc[saopaulo_merged['Cluster Labels'] == 2, 
                     saopaulo_merged.columns[[0] + list(range(4, saopaulo_merged.shape[1]))]]         # Examine cluster 4
saopaulo_merged.loc[saopaulo_merged['Cluster Labels'] == 3, 
                     saopaulo_merged.columns[[0] + list(range(4, saopaulo_merged.shape[1]))]]         # Examine cluster 5
saopaulo_merged.loc[saopaulo_merged['Cluster Labels'] == 4, 
                     saopaulo_merged.columns[[0] + list(range(4, saopaulo_merged.shape[1]))]]   
# Examine cluster 6
saopaulo_merged.loc[saopaulo_merged['Cluster Labels'] == 5, 
                     saopaulo_merged.columns[[0] + list(range(4, saopaulo_merged.shape[1]))]]
# Examine cluster 7
saopaulo_merged.loc[saopaulo_merged['Cluster Labels'] == 6, 
                     saopaulo_merged.columns[[0] + list(range(4, saopaulo_merged.shape[1]))]]
```
Then it returns dataframe like this as an example:
![Image Description](https://i.postimg.cc/YCvhcp0c/13-Examine-cluster.png)


- Create a Dataframes for each clusters with number of 1st Most Common Venue counted to create a bar graph
```
# Count venues for cluster 0 and create a dataframe
df_cluster_0 = pd.DataFrame(columns=['Cluster', 'Count', '1st Most Common Venue'])
count0r = cluster_1st[(cluster_1st['Cluster Labels']==0) & (cluster_1st['1st Most Common Venue']=='Restaurant')].count()
count0g = cluster_1st[(cluster_1st['Cluster Labels']==0) & (cluster_1st['1st Most Common Venue']=='Grocery')].count()

df_cluster_0.loc[0, 'Cluster'] = 0
df_cluster_0.loc[1, 'Cluster'] = 0
    
df_cluster_0.loc[0,'Count'] = count0r['Cluster Labels']
df_cluster_0.loc[1, 'Count'] = count0g['Cluster Labels']
df_cluster_0.loc[0,'1st Most Common Venue'] = 'Restaurant'
df_cluster_0.loc[1,'1st Most Common Venue'] = 'Grocery'


# Count venues for cluster 1 and create a dataframe
df_cluster_1 = pd.DataFrame(columns=['Cluster', 'Count', '1st Most Common Venue'])
count1r = cluster_1st[(cluster_1st['Cluster Labels']==1) & (cluster_1st['1st Most Common Venue']=='Restaurant')].count()

df_cluster_1.loc[0, 'Cluster'] = 1

df_cluster_1.loc[0,'Count'] = count1r['Cluster Labels']
df_cluster_1.loc[0,'1st Most Common Venue'] = 'Restaurant'


# Count venues for cluster 2 and create a dataframe
df_cluster_2 = pd.DataFrame(columns=['Cluster', 'Count', '1st Most Common Venue'])
count2r = cluster_1st[(cluster_1st['Cluster Labels']==2) & (cluster_1st['1st Most Common Venue']=='Restaurant')].count()
count2m = cluster_1st[(cluster_1st['Cluster Labels']==2) & (cluster_1st['1st Most Common Venue']=='Mall')].count()
count2h = cluster_1st[(cluster_1st['Cluster Labels']==2) & (cluster_1st['1st Most Common Venue']=='Home Store')].count()
count2gm = cluster_1st[(cluster_1st['Cluster Labels']==2) & (cluster_1st['1st Most Common Venue']=='Gallery & Museum')].count()
count2d = cluster_1st[(cluster_1st['Cluster Labels']==2) & (cluster_1st['1st Most Common Venue']=='Dessert')].count()

df_cluster_2.loc[0, 'Cluster'] = 2
df_cluster_2.loc[1, 'Cluster'] = 2
df_cluster_2.loc[2, 'Cluster'] = 2
df_cluster_2.loc[3, 'Cluster'] = 2
df_cluster_2.loc[4, 'Cluster'] = 2

df_cluster_2.loc[0,'Count'] = count2r['Cluster Labels']
df_cluster_2.loc[1,'Count'] = count2m['Cluster Labels']
df_cluster_2.loc[2,'Count'] = count2h['Cluster Labels']
df_cluster_2.loc[3,'Count'] = count2gm['Cluster Labels']
df_cluster_2.loc[4,'Count'] = count2d['Cluster Labels']

df_cluster_2.loc[0,'1st Most Common Venue'] = 'Restaurant'
df_cluster_2.loc[1,'1st Most Common Venue'] = 'Mall'
df_cluster_2.loc[2,'1st Most Common Venue'] = 'Home Store'
df_cluster_2.loc[3,'1st Most Common Venue'] = 'Gallery & Museum'
df_cluster_2.loc[4,'1st Most Common Venue'] = 'Dessert'


# Count venues for cluster 3 and create a dataframe
df_cluster_3 = pd.DataFrame(columns=['Cluster', 'Count', '1st Most Common Venue'])

df_cluster_3.loc[0, 'Cluster'] = 3

count3r = cluster_1st[(cluster_1st['Cluster Labels']==3) & (cluster_1st['1st Most Common Venue']=='Restaurant')].count()

df_cluster_3.loc[0,'Count'] = count3r['Cluster Labels']
df_cluster_3.loc[0,'1st Most Common Venue'] = 'Restaurant'


# Count venues for cluster 4 and create a dataframe
df_cluster_4 = pd.DataFrame(columns=['Cluster', 'Count', '1st Most Common Venue'])

df_cluster_4.loc[0, 'Cluster'] = 4

count4pb = cluster_1st[(cluster_1st['Cluster Labels']==4) & (cluster_1st['1st Most Common Venue']=='Pastry & Bakery')].count()
df_cluster_4.loc[0,'Count'] = count4pb['Cluster Labels']
df_cluster_4.loc[0,'1st Most Common Venue'] = 'Pastry & Bakery'


# Count venues for cluster 5 and create a dataframe
df_cluster_5 = pd.DataFrame(columns=['Cluster', 'Count', '1st Most Common Venue'])

df_cluster_5.loc[0, 'Cluster'] = 5
df_cluster_5.loc[1, 'Cluster'] = 5
df_cluster_5.loc[2, 'Cluster'] = 5

count5pb = cluster_1st[(cluster_1st['Cluster Labels']==5) & (cluster_1st['1st Most Common Venue']=='Pastry & Bakery')].count()
count5m = cluster_1st[(cluster_1st['Cluster Labels']==5) & (cluster_1st['1st Most Common Venue']=='Mall')].count()
count5g = cluster_1st[(cluster_1st['Cluster Labels']==5) & (cluster_1st['1st Most Common Venue']=='Grocery')].count()

df_cluster_5.loc[0,'Count'] = count5pb['Cluster Labels']
df_cluster_5.loc[0,'1st Most Common Venue'] = 'Pastry & Bakery'
df_cluster_5.loc[1, 'Count'] = count5m['Cluster Labels']
df_cluster_5.loc[1, '1st Most Common Venue'] = 'Mall'
df_cluster_5.loc[2, 'Count'] = count5g['Cluster Labels']
df_cluster_5.loc[2, '1st Most Common Venue'] = 'Grocery'


# Count venues for cluster 6 and create a dataframe
df_cluster_6 = pd.DataFrame(columns=['Cluster', 'Count', '1st Most Common Venue'])

df_cluster_6.loc[0, 'Cluster'] = 6
df_cluster_6.loc[1, 'Cluster'] = 6


count6r = cluster_1st[(cluster_1st['Cluster Labels']==6) & (cluster_1st['1st Most Common Venue']=='Restaurant')].count()
count6pb = cluster_1st[(cluster_1st['Cluster Labels']==6) & (cluster_1st['1st Most Common Venue']=='Pastry & Bakery')].count()

df_cluster_6.loc[0,'Count'] = count6r['Cluster Labels']
df_cluster_6.loc[0,'1st Most Common Venue'] = 'Restaurant'
df_cluster_6.loc[1,'Count'] = count6pb['Cluster Labels']
df_cluster_6.loc[1,'1st Most Common Venue'] = 'Pastry & Bakery'
```







2. What is the most popular products? (Which product is creating revenue the most?)


3. The month with the highest revenues (sales prediction)


4. Customer satisfaction with products(1,2: unsatisfied, 3: moderate, 4,5: satisfied). --> Product Quality 


5. Delivery Performance (Consumption of time till the arrival of the parcels)




6. Who pays more for the delivery fee? (Depends on city and purchase cost)



Elbow method to determine the optimal number of clusters for K-means clustering
