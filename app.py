import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    file  = 'env/aircrahesFullDataUpdated_2024.csv'

    df = pd.read_csv(file)
    return df

df = load_data()
df.head()
df.info()

# Data cleaning
df['Country/Region'] = df['Country/Region'].replace('-', pd.NA)
df['Country/Region'] = df['Country/Region'].fillna('Unknown')
df['Operator'] = df['Operator'].fillna('Unknown')
df['Country/Region'] = df['Country/Region'].str.strip().str.title()
df['Aircraft Manufacturer'] = df['Aircraft Manufacturer'].str.strip().str.title()
df['Aircraft'] = df['Aircraft'].str.strip()
df['Location'] = df['Location'].str.strip()
df['Operator'] = df['Operator'].str.strip()
df['Quarter'] = pd.Categorical(df['Quarter'], categories=['Qtr 1', 'Qtr 2', 'Qtr 3', 'Qtr 4'])
df['Month'] = pd.Categorical(df['Month'], categories=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'])

# Handling outliers
Q1_ground = df['Ground'].quantile(0.25)
Q3_ground = df['Ground'].quantile(0.75)
IQR_ground = Q3_ground - Q1_ground

Q1_fatalities = df['Fatalities (air)'].quantile(0.25)
Q3_fatalities = df['Fatalities (air)'].quantile(0.75)
IQR_fatalities = Q3_fatalities - Q1_fatalities

lower_bound_ground = Q1_ground - 1.5 * IQR_ground
upper_bound_ground = Q3_ground + 1.5 * IQR_ground

lower_bound_fatalities = Q1_fatalities - 1.5 * IQR_fatalities
upper_bound_fatalities = Q3_fatalities + 1.5 * IQR_fatalities

df['Ground'] = df['Ground'].clip(lower=lower_bound_ground, upper=upper_bound_ground)
df['Fatalities (air)'] = df['Fatalities (air)'].clip(lower=lower_bound_fatalities, upper=upper_bound_fatalities)

duplicate_rows = df.duplicated().sum()
df.drop_duplicates(inplace=True)

# Streamlit app
st.title('Air Crashes Overview')

# Sidebar filters
st.sidebar.header('Filter the data')
selected_year = st.sidebar.selectbox('Select Year(s)', options=['All Years'] + list(df['Year'].unique()), index=0)

if selected_year == 'All Years':
    filtered_df = df
else:
    filtered_df = df[df['Year'] == selected_year]

# Top Aircraft Manufacturers by Number of Crashes
st.header('Top Aircraft Manufacturers by Number of Crashes')
Aircraft_Manufacturer = filtered_df['Aircraft Manufacturer'].value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.barh(Aircraft_Manufacturer.index, Aircraft_Manufacturer.values, color='skyblue')
plt.xlabel('Number of Crashes')
plt.ylabel('Aircraft Manufacturer')
plt.title('Top 10 Aircraft Manufacturers with the Highest Number of Crashes')
plt.gca().invert_yaxis()
st.pyplot(plt)

# Air Crash Trends Over the Years
yearly_crash_count = df.groupby('Year').size().reset_index(name='Crash Count')
st.header('Air Crash Trends Over the Years')
plt.figure(figsize=(12, 6))
plt.plot(yearly_crash_count['Year'], yearly_crash_count['Crash Count'], color='blue', marker='o')
plt.xlabel('Year')
plt.ylabel('Number of Crashes')
plt.title('Trend of Air Crashes Over the Years')
plt.grid(True)
st.pyplot(plt)

# Countries/Regions with the Highest Number of Air Crash Fatalities
fatalities_by_country = df.groupby('Country/Region')['Fatalities (air)'].sum().reset_index()
top_10_countries_fatalities = fatalities_by_country.sort_values(by='Fatalities (air)', ascending=False).head(10)
st.header('Countries/Regions with the Highest Number of Air Crash Fatalities')
plt.figure(figsize=(10, 6))
plt.barh(top_10_countries_fatalities['Country/Region'], top_10_countries_fatalities['Fatalities (air)'], color='coral')
plt.xlabel('Number of Fatalities')
plt.ylabel('Country/Region')
plt.title('Top 10 Countries/Regions with the Highest Number of Fatalities')
plt.gca().invert_yaxis()
st.pyplot(plt)

# Air Crashes by Quarter of the Year
crashes_by_quarter = df.groupby('Quarter', observed=True).size().reset_index(name='Crash Count')
st.header('Air Crashes by Quarter of the Year')
plt.figure(figsize=(8, 5))
plt.bar(crashes_by_quarter['Quarter'], crashes_by_quarter['Crash Count'], color='lightgreen')
plt.xlabel('Quarter')
plt.ylabel('Number of Crashes')
plt.title('Air Crashes by Quarter')
st.pyplot(plt)

# Relationship Between Number of People Aboard and Air Crash Fatalities
st.header('Relationship Between Number of People Aboard and Air Crash Fatalities')
plt.figure(figsize=(8, 6))
plt.scatter(df['Aboard'], df['Fatalities (air)'], color='red', alpha=0.5)
plt.xlabel('Number of People Aboard')
plt.ylabel('Fatalities (air)')
plt.title('Relationship Between People Aboard and Air Fatalities')
plt.grid(True)
st.pyplot(plt)
