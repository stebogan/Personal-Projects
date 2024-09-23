import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Variable dictionary


# Load unprocessed data
booking_data = pd.read_csv("hotel_booking.csv")
columns = booking_data.columns

# Look at preliminary statistics of data set
stats = booking_data.describe()
data_means = booking_data.groupby("is_canceled").mean(numeric_only=True)
data_stds = booking_data.groupby("is_canceled").std(numeric_only=True)
canceled_data= booking_data[booking_data['is_canceled']==1]
non_canceled_data= booking_data[booking_data['is_canceled']==0]

# investigate and remove null values
print(booking_data.isnull().sum(axis=0))
booking_data = booking_data.fillna(0)


# Remove error country code 0
booking_data = booking_data.drop(booking_data[booking_data['country'] == 0].index)

# Remove error number of adults = 0
booking_data = booking_data.drop(booking_data[booking_data['adults'] == 0].index)

#%%Visualizations

# Visualize number of adults    
plt.figure(figsize=(14,8))
plt.hist(booking_data['adults'],bins=50)
plt.xticks(booking_data['adults'].unique())
plt.yscale('log')
plt.title('Distribution of adults')
plt.ylabel('log count')
plt.xlabel('# adults')
plt.show()
    
# % total bookings canceled by country

top_10_country = canceled_data['country'].value_counts()[:10]/booking_data['country'].value_counts()[canceled_data['country'].value_counts()[:10].index]
plt.figure(figsize=(8, 8))  # Set background color to a light brown
plt.title('Top 10 countries with % total bookings canceled', color="black")
plt.pie(top_10_country.sort_values(),autopct='%.2f', labels=top_10_country.sort_values().index)
plt.show()

# Correlation matrix
corr = booking_data.select_dtypes(include='number').corr()

fig, ax = plt.subplots(figsize=(14, 8))
ax.matshow(corr, cmap='terrain')
axis=np.arange(len(corr.index.values.tolist()))
ax.set_xticks(axis)
ax.tick_params(axis='x', labelrotation=90)
ax.set_yticks(axis)
ax.set_xticklabels(corr.index.values.tolist())
ax.set_yticklabels(corr.index.values.tolist())

for (i, j), z in np.ndenumerate(corr):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',fontsize='x-small',color='black')
plt.show()

# Canceled/non_canceled histograms
# Lead time
plt.figure(figsize=(14,8))
plt.hist(canceled_data['lead_time'],bins=40, alpha=0.4, label='canceled', color='red')
plt.hist(non_canceled_data['lead_time'],bins=40, alpha=0.4, label='non-canceled', color='green')
plt.text(100, 12000, 'Canceled: mean = {0} | std = {1}\nNon-canceled: mean = {2} | std = {3}'.format(round(data_means["lead_time"][1],2),round(data_stds["lead_time"][1],2),round(data_means["lead_time"][0],2),round(data_stds["lead_time"][0],2)), fontsize = 12)
plt.legend(loc='upper right')
plt.title("Lead Time [days]")
plt.xlabel("lead time [days]")
plt.ylabel("count")
plt.yscale('log')
plt.show()

# Days in wait list
plt.figure(figsize=(14,8))
plt.hist(canceled_data['days_in_waiting_list'],bins=70, alpha=0.4, label='cancelled', color='red')
plt.hist(non_canceled_data['days_in_waiting_list'],bins=70, alpha=0.4, label='non-cancelled', color='green')
plt.text(100, 25000, 'Canceled: mean = {0} | std = {1}\nNon-canceled: mean = {2} | std = {3}'.format(round(data_means["days_in_waiting_list"][1],2),round(data_stds["days_in_waiting_list"][1],2),round(data_means["days_in_waiting_list"][0],2),round(data_stds["days_in_waiting_list"][0],2)), fontsize = 12)
plt.legend(loc='upper right')
plt.title("Days in waiting list [days]")
plt.xlabel("wait time [days]")
plt.ylabel("count")
plt.yscale('log')
plt.show()

#Number of booking changes
plt.figure(figsize=(14,8))
plt.hist(canceled_data['booking_changes'],bins=22, alpha=0.4, label='cancelled', color='red')
plt.hist(non_canceled_data['booking_changes'],bins=22, alpha=0.4, label='non-cancelled', color='green')
plt.text(12, 10000, 'Canceled: mean = {0} | std = {1}\nNon-canceled: mean = {2} | std = {3}'.format(round(data_means["booking_changes"][1],2),round(data_stds["booking_changes"][1],2),round(data_means["booking_changes"][0],2),round(data_stds["booking_changes"][0],2)), fontsize = 12)
plt.legend(loc='upper right')
plt.title("Number of Booking Changes")
plt.xlabel("number of changes")
plt.ylabel("count")
plt.yscale('log')
plt.show()

#Arrival date week number
plt.figure(figsize=(14,8))
plt.hist(canceled_data['arrival_date_week_number'],bins=52, alpha=0.4, label='cancelled', color='red')
plt.hist(non_canceled_data['arrival_date_week_number'],bins=52, alpha=0.4, label='non-cancelled', color='green')
plt.text(12, 10000, 'Canceled: mean = {0} | std = {1}\nNon-canceled: mean = {2} | std = {3}'.format(round(data_means["booking_changes"][1],2),round(data_stds["booking_changes"][1],2),round(data_means["booking_changes"][0],2),round(data_stds["booking_changes"][0],2)), fontsize = 12)
plt.legend(loc='upper right')
plt.title("Arrival Date Week Number")
plt.xlabel("week number")
plt.ylabel("count")
plt.yscale('log')
plt.show()

#%% Regression analysis

# Remove unneccessary columns (low correlation, data leak, too specific)
booking_data = booking_data.drop(['agent','company','reservation_status','reservation_status_date','name','email','phone-number','credit_card','arrival_date_year'], axis=1)
columns = booking_data.columns

 

cat_var = [0,3,11,13,14,18,19,21,23]
num_var = [2,4,5,6,7,8,9,10,15,16,17,20,22,24,25,26]

cat_df = booking_data.iloc[:,cat_var]

# encode categorical variables
cat_df['hotel'] = cat_df['hotel'].map({'Resort Hotel' : 0, 'City Hotel' : 1})
cat_df['arrival_date_month'] = cat_df['arrival_date_month'].map({'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6, 'August':7, 'September':8, 'October':9, 'November':10, 'December':11})
cat_df['meal'] = cat_df['meal'].map({'BB' : 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})
cat_df['market_segment'] = cat_df['market_segment'].map({'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7})
cat_df['distribution_channel'] = cat_df['distribution_channel'].map({'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3,'GDS': 4})
cat_df['reserved_room_type'] = cat_df['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6,'L': 7, 'B': 8})
cat_df['assigned_room_type'] = cat_df['assigned_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6,'L': 7, 'B': 8, 'I': 9, 'K': 10})
cat_df['deposit_type'] = cat_df['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3})
cat_df['customer_type'] = cat_df['customer_type'].map({'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3})

# normalize numeric dataframe
num_df = booking_data.iloc[:,num_var]  
num_df = num_df/num_df.max(axis=0)

# Recombine data to form features and target set
X = pd.concat([cat_df, num_df], axis = 1)
y = booking_data['is_canceled']

print(cat_df.isnull().sum(axis=0))
#booking_data = booking_data.fillna(0)

# Train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) 


#%% Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
    
model1 = RandomForestClassifier(n_estimators=50)
model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

print(f"Accuracy Score of Random Forest Classifier is : {accuracy}")
print(f"Confusion Matrix : \n{conf}")
