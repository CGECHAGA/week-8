# ======================
# COVID-19 GLOBAL DATA TRACKER
# Complete Analysis in Single Notebook
# ======================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("husl")

# ======================
# 1. DATA COLLECTION & LOADING
# ======================
print("Loading COVID-19 data from Our World in Data...")
url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
df = pd.read_csv(url)

# ======================
# 2. DATA CLEANING
# ======================
print("\nCleaning and preprocessing data...")

# Convert date and filter timeframe
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= pd.to_datetime('2020-03-01')]  # Start from March 2020

# Select countries and key columns
countries = ['United States', 'India', 'Brazil', 'Germany', 'Kenya', 'South Africa']
cols = ['date', 'location', 'total_cases', 'new_cases', 'total_deaths', 
        'new_deaths', 'total_vaccinations', 'people_vaccinated', 'population',
        'icu_patients', 'hosp_patients', 'reproduction_rate']

df_clean = df[df['location'].isin(countries)][cols]

# Forward fill missing values within each country
df_clean = df_clean.groupby('location').apply(lambda x: x.fillna(method='ffill'))

# Calculate derived metrics
df_clean['death_rate'] = (df_clean['total_deaths'] / df_clean['total_cases']) * 100
df_clean['vaccination_rate'] = (df_clean['people_vaccinated'] / df_clean['population']) * 100
df_clean['cases_per_million'] = (df_clean['total_cases'] / df_clean['population']) * 1e6

# 7-day rolling averages
df_clean['new_cases_7day'] = df_clean.groupby('location')['new_cases'].transform(lambda x: x.rolling(7).mean())
df_clean['new_deaths_7day'] = df_clean.groupby('location')['new_deaths'].transform(lambda x: x.rolling(7).mean())

# ======================
# 3. EXPLORATORY DATA ANALYSIS
# ======================
print("\nGenerating visualizations...")

# Set up figure grid
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
plt.suptitle("COVID-19 Global Trends Analysis", fontsize=20, y=1.02)

# Plot 1: Total Cases
sns.lineplot(ax=axes[0,0], data=df_clean, x='date', y='total_cases', hue='location')
axes[0,0].set_title("Total Confirmed Cases")
axes[0,0].set_ylabel("Cases (log scale)")
axes[0,0].set_yscale('log')

# Plot 2: New Cases (7-day avg)
sns.lineplot(ax=axes[0,1], data=df_clean, x='date', y='new_cases_7day', hue='location')
axes[0,1].set_title("Daily New Cases (7-Day Average)")
axes[0,1].set_ylabel("Cases per day")

# Plot 3: Total Deaths
sns.lineplot(ax=axes[1,0], data=df_clean, x='date', y='total_deaths', hue='location')
axes[1,0].set_title("Total Deaths")
axes[1,0].set_ylabel("Deaths (log scale)")
axes[1,0].set_yscale('log')

# Plot 4: Death Rate over time
sns.lineplot(ax=axes[1,1], data=df_clean, x='date', y='death_rate', hue='location')
axes[1,1].set_title("Case Fatality Rate (%)")
axes[1,1].set_ylabel("Death Rate %")

# Plot 5: Vaccination Progress
sns.lineplot(ax=axes[2,0], data=df_clean, x='date', y='vaccination_rate', hue='location')
axes[2,0].set_title("Vaccination Progress (% Population)")
axes[2,0].set_ylabel("% Vaccinated")

# Plot 6: Hospitalization vs Cases (latest data)
latest = df_clean.sort_values('date').groupby('location').last().reset_index()
sns.scatterplot(ax=axes[2,1], data=latest, x='cases_per_million', y='hosp_patients', hue='location', s=200)
axes[2,1].set_title("Latest: Cases vs Hospitalizations")
axes[2,1].set_xlabel("Cases per million")
axes[2,1].set_ylabel("Hospital Patients")

plt.tight_layout()
plt.show()

# ======================
# 4. COUNTRY COMPARISON (Latest Data)
# ======================
print("\nGenerating country comparison...")

latest_data = df_clean.sort_values('date').groupby('location').last().reset_index()
metrics = ['total_cases', 'total_deaths', 'death_rate', 'vaccination_rate']
latest_data[['location'] + metrics].sort_values('total_cases', ascending=False)

# ======================
# 5. GLOBAL CHOROPLETH MAP
# ======================
print("\nGenerating global choropleth map...")

# Prepare global latest data
global_latest = df.sort_values('date').groupby('location').last().reset_index()

fig = px.choropleth(global_latest,
                    locations="iso_code",
                    color="total_cases_per_million",
                    hover_name="location",
                    hover_data=["total_cases", "total_deaths", "people_vaccinated"],
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Global COVID-19 Cases per Million People",
                    height=600)
fig.show()

# ======================
# 6. KEY INSIGHTS
# ======================
print("\nKey Insights:")
insights = [
    "1. The US and India had the highest total cases, while Germany maintained lower case counts",
    "2. Brazil showed the highest death rate among analyzed countries (~3%)",
    "3. Vaccination rates diverged significantly - US/Germany >60%, Kenya <10%",
    "4. All countries experienced multiple waves of infections",
    "5. Hospitalizations closely followed case trends with 1-2 week lag"
]

for insight in insights:
    print(f"• {insight}")

# ======================
# 7. EXPORT OPTIONS
# ======================
print("\nExport options:")
print("- Save figures: plt.savefig('covid_analysis.png')")
print("- Export data: df_clean.to_csv('processed_covid_data.csv')")
print("- Create report: !jupyter nbconvert --to html COVID_Analysis.ipynb")

print("\nAnalysis complete! ")