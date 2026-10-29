import math
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='Toftir Weather AI', page_icon='🌦️', layout='wide')
st.title('🌦️ Toftir Weather AI')
st.caption('Høganesvegur 87, 650 Toftir • Multi-model weather outlook • Not a site measurement')

# Approximate Toftir centre. Replace with verified address coordinates in sidebar.
DEFAULT_LAT = 62.097
DEFAULT_LON = -6.735
MODELS = {
    'DMI HARMONIE': 'dmi_harmonie_arome_europe',
    'ECMWF IFS': 'ecmwf_ifs',
    'NOAA GFS': 'ncep_gfs_global',
    'DWD ICON': 'icon_global',
}
VARIABLES = 'wind_speed_10m,wind_direction_10m,precipitation,temperature_2m'
COMPASS = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']

def frost_risk(temp_c, rain_mm):
    """Simple forecast screening, NOT a measured road-surface ice probability."""
    if temp_c <= 0:
        return 'Air frost forecast'
    if temp_c <= 2 and rain_mm > 0:
        return 'Possible slippery surfaces'
    if temp_c <= 2:
        return 'Near freezing'
    return 'Low temperature-related risk'


def compass(deg):
    return COMPASS[round(float(deg) / 22.5) % 16]


def wind_arrow(deg):
    arrows = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖']
    return arrows[int((float(deg) % 360) / 45 + 0.5) % 8]

def circular_mean_deg(dirs):
    radians = np.radians(np.asarray(dirs, dtype=float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))) % 360)

def circular_spread_deg(dirs, centre):
    diffs = (np.asarray(dirs, dtype=float) - centre + 180) % 360 - 180
    return float(np.sqrt(np.mean(diffs ** 2)))

def plot_daily_wind_roses(frame):
    """Plot one polar wind rose per local forecast day.

    Bars count hourly forecast samples. Angle represents wind FROM direction.
    This visualization is not measured climate history.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    speed_edges = [0, 3, 6, 9, 12, 16, 21, float('inf')]
    speed_labels = ['0–3', '3–6', '6–9', '9–12', '12–16', '16–21', '21+']
    colors = ['#c8e2e2', '#74b6a7', '#389e79', '#bedb63', '#f4cb45', '#ee9341', '#bd4549']
    sectors = 16
    groups = list(frame.groupby(frame['Time'].dt.date, sort=True))
    n = len(groups)
    fig, axes = plt.subplots(1, n, subplot_kw={'projection': 'polar'}, figsize=(max(4.5, 3.5*n), 4.6))
    if n == 1:
        axes = [axes]
    max_n = 1
    daily_hist = []
    for _, day in groups:
        hist = np.zeros((sectors, len(speed_labels)), dtype=int)
        for _, row in day.iterrows():
            direction_idx = int((float(row['Direction (°)']) + 11.25) // 22.5) % 16
            speed_idx = int(np.searchsorted(speed_edges[1:], float(row['Wind (m/s)']), side='right'))
            hist[direction_idx, speed_idx] += 1
        daily_hist.append(hist)
        max_n = max(max_n, int(hist.sum(axis=1).max()))

    for ax, ((date, day), hist) in zip(axes, zip(groups, daily_hist)):
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        angles = np.arange(sectors) * 2 * np.pi / sectors
        bottom = np.zeros(sectors)
        for i, color in enumerate(colors):
            heights = hist[:, i]
            ax.bar(angles, heights, width=2*np.pi/sectors*0.88,
                   bottom=bottom, color=color, edgecolor='white', linewidth=0.25)
            bottom += heights
        ax.set_ylim(0, max_n + 0.5)
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
        ax.set_xticklabels(['N','NE','E','SE','S','SW','W','NW'], fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(alpha=0.4)
        ax.set_title(f"{date:%a %d %b}\n{len(day)} forecast hours", fontsize=10, pad=16)

    fig.legend(handles=[Patch(facecolor=c, label=f'{lab} m/s') for c, lab in zip(colors, speed_labels)],
               loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=min(7, len(speed_labels)), fontsize=8,
               frameon=False)
    fig.subplots_adjust(bottom=0.22, top=0.82, left=0.03, right=0.97, wspace=0.35)
    return fig

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_model(model, lat, lon):
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': lat, 'longitude': lon, 'models': model,
        'hourly': VARIABLES, 'wind_speed_unit': 'ms',
        'precipitation_unit': 'mm', 'timezone': 'Atlantic/Faroe',
        'forecast_hours': 72, 'cell_selection': 'land'
    }
    response = requests.get(url, params=params, timeout=35)
    response.raise_for_status()
    result = response.json()
    hourly = result['hourly']
    suffix = '_' + model
    def get(name):
        # One model per call normally yields unsuffixed keys; handle both formats.
        key = name if name in hourly else name + suffix
        return hourly[key]
    frame = pd.DataFrame({
        'time': pd.to_datetime(hourly['time']),
        'speed': get('wind_speed_10m'),
        'direction': get('wind_direction_10m'),
        'rain': get('precipitation'),
        'temp': get('temperature_2m'),
    })
    return frame.dropna()

with st.sidebar:
    st.header('Location & settings')
    lat = st.number_input('Latitude (WGS84)', value=DEFAULT_LAT, format='%.6f')
    lon = st.number_input('Longitude (WGS84)', value=DEFAULT_LON, format='%.6f')
    st.info('Starting coordinates are approximate Toftir coordinates, NOT verified for Høganesvegur 87. Replace them before relying on a site-specific forecast.')
    hours = st.slider('Hours to display', 12, 72, 48, 12)
    if st.button('Refresh forecast'):
        fetch_model.clear()

available = {}
errors = []
with st.spinner('Collecting forecasts from weather models…'):
    for label, code in MODELS.items():
        try:
            df = fetch_model(code, lat, lon)
            if not df.empty:
                available[label] = df.set_index('time')
        except (requests.RequestException, KeyError, ValueError, TypeError) as error:
            errors.append(f'{label}: {error}')

if not available:
    st.error('Unable to retrieve weather forecasts. Check your network connection and the model API availability.')
    for error in errors:
        st.caption(error)
    st.stop()

if errors:
    st.warning('Some models are unavailable. Their forecasts were excluded: ' + ' | '.join(errors))

# Only combine times for which at least two models have complete data.
records = []
all_times = sorted(set().union(*[set(frame.index) for frame in available.values()]))
for t in all_times:
    entries = []
    for model, frame in available.items():
        if t in frame.index:
            row = frame.loc[t]
            if all(pd.notna(row[col]) for col in ('speed','direction','rain','temp')):
                entries.append((model, float(row['speed']), float(row['direction']), float(row['rain']), float(row['temp'])))
    if len(entries) < 2:
        continue
    speeds = np.asarray([item[1] for item in entries])
    directions = np.asarray([item[2] for item in entries])
    rain = np.asarray([item[3] for item in entries])
    temps = np.asarray([item[4] for item in entries])
    direction = circular_mean_deg(directions)
    records.append({
        'Time': t, 'Wind (m/s)': float(np.mean(speeds)),
        'Direction (°)': round(direction), 'Compass': compass(direction),
        'Rain (mm/h)': float(np.mean(rain)),
        'Temperature (°C)': float(np.mean(temps)),
        'Temperature spread (°C)': float(np.std(temps)),
        'Cold-model range (°C)': float(np.min(temps)),
        'Wind spread (m/s)': float(np.std(speeds)),
        'Direction spread (°)': circular_spread_deg(directions, direction),
        'Rain spread (mm/h)': float(np.std(rain)),
        'Models': len(entries),
    })

if not records:
    st.error('Not enough overlapping forecast data to calculate a combined forecast.')
    st.stop()

forecast = pd.DataFrame(records)
forecast['Wind arrow'] = forecast['Direction (°)'].apply(wind_arrow)
forecast['Direction'] = forecast.apply(lambda row: f"{row['Wind arrow']} {row['Compass']} ({int(row['Direction (°)'])}°)", axis=1)
forecast['Freezing risk'] = [
    frost_risk(t, r) for t, r in zip(forecast['Temperature (°C)'], forecast['Rain (mm/h)'])
]
now = pd.Timestamp.now(tz='Atlantic/Faroe').tz_localize(None)
forecast = forecast[forecast['Time'] >= now.floor('h')].head(hours).reset_index(drop=True)
if forecast.empty:
    st.error('No future hourly forecasts were returned.')
    st.stop()

first = forecast.iloc[0]
a, b, c, d, e = st.columns(5)
a.metric('Wind', f"{first['Wind (m/s)']:.1f} m/s")
b.metric('Direction', first['Direction'])
c.metric('Rain', f"{first['Rain (mm/h)']:.1f} mm/h")
d.metric('Temperature', f"{first['Temperature (°C)']:.1f} °C")
e.metric('Freezing risk', first['Freezing risk'])
st.caption(f"Models available: {first['Models']}/{len(MODELS)}")

st.subheader('Daily wind roses')
st.caption('Each rose shows the number of forecast hours with wind FROM each direction; colors indicate forecast wind speed (m/s). Partial days are labelled by the number of included hours. These are forecasts, not observations.')
rose_fig = plot_daily_wind_roses(forecast)
st.pyplot(rose_fig, use_container_width=True)
import matplotlib.pyplot as plt
plt.close(rose_fig)

def hourly_chart(data, columns, ylabel):
    """Render chart with an explicit Faroese-local 24-hour time axis."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(11, 3.2))
    for name in columns:
        ax.plot(data['Time'], data[name], label=name, linewidth=1.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Local Faroe Islands time (24-hour clock)')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize=9)
    fig.autofmt_xdate(rotation=25, ha='right')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.subheader('Combined hourly forecast')
hourly_chart(forecast, ['Wind (m/s)', 'Wind spread (m/s)'], 'm/s')
hourly_chart(forecast, ['Rain (mm/h)', 'Rain spread (mm/h)'], 'mm/h')
st.subheader('Temperature and freezing risk')
hourly_chart(forecast, ['Temperature (°C)', 'Temperature spread (°C)'], '°C')
st.caption('Freezing risk is a cautious screening label based on forecast air temperature and precipitation, not a probability of ice. Roads, bridges, and exposed surfaces can freeze while air temperature is above 0°C.')

st.subheader('Hourly forecast and uncertainty')
display_forecast = forecast.copy()
display_forecast['Time'] = display_forecast['Time'].dt.strftime('%a %d %b %H:%M')
display_forecast = display_forecast[[
    'Time', 'Direction', 'Wind (m/s)', 'Wind spread (m/s)',
    'Rain (mm/h)', 'Rain spread (mm/h)',
    'Temperature (°C)', 'Temperature spread (°C)', 'Cold-model range (°C)',
    'Direction spread (°)', 'Models', 'Freezing risk'
]]
st.dataframe(display_forecast.style.format({
    'Wind (m/s)': '{:.1f}', 'Rain (mm/h)': '{:.2f}',
    'Wind spread (m/s)': '{:.1f}', 'Direction spread (°)': '{:.0f}',
    'Rain spread (mm/h)': '{:.2f}',
    'Temperature (°C)': '{:.1f}', 'Temperature spread (°C)': '{:.1f}',
    'Cold-model range (°C)': '{:.1f}'
}), hide_index=True, use_container_width=True)

st.subheader('What the forecasts suggest')
worst_wind = forecast.loc[forecast['Wind (m/s)'].idxmax()]
wettest = forecast.loc[forecast['Rain (mm/h)'].idxmax()]
st.write(f"Strongest combined hourly wind: **{worst_wind['Wind (m/s)']:.1f} m/s** "
         f"from **{worst_wind['Compass']}** around **{worst_wind['Time']:%a %d %b %H:%M}**.")
st.write(f"Wettest forecast hour: **{wettest['Rain (mm/h)']:.2f} mm** "
         f"around **{wettest['Time']:%a %d %b %H:%M}**.")

coldest = forecast.loc[forecast['Temperature (°C)'].idxmin()]
st.write(f"Lowest combined air temperature: **{coldest['Temperature (°C)']:.1f} °C** "
         f"around **{coldest['Time']:%a %d %b %H:%M}**, "
         f"with screening label **{coldest['Freezing risk']}**.")
st.warning('Do not use this forecast-only freezing indicator as an ice safety guarantee. '
           'It does not include ground/road surface temperatures, humidity, radiation, or treatment conditions.')

st.subheader('Individual forecast models')
chosen = st.selectbox('Select model', list(available))
model_df = available[chosen].reset_index()
model_df = model_df.head(hours).rename(columns={
    'time': 'Time', 'speed': 'Wind (m/s)', 'direction': 'Wind from (°)', 'rain': 'Rain (mm/h)', 'temp': 'Temperature (°C)'
})
model_df['Wind arrow'] = model_df['Wind from (°)'].apply(wind_arrow)
model_df['Compass'] = model_df['Wind from (°)'].apply(compass)
model_df['Direction'] = model_df.apply(lambda row: f"{row['Wind arrow']} {row['Compass']} ({int(round(row['Wind from (°)']))}°)", axis=1)
model_df['Time'] = pd.to_datetime(model_df['Time']).dt.strftime('%a %d %b %H:%M')
model_df = model_df[['Time', 'Direction', 'Wind (m/s)', 'Rain (mm/h)', 'Temperature (°C)', 'Wind from (°)']]
st.dataframe(model_df.style.format({
    'Wind (m/s)': '{:.1f}', 'Rain (mm/h)': '{:.2f}', 'Temperature (°C)': '{:.1f}', 'Wind from (°)': '{:.0f}'
}), hide_index=True, use_container_width=True)

st.download_button('Download combined forecast as CSV', forecast.assign(Time=forecast['Time'].dt.strftime('%Y-%m-%d %H:%M')).to_csv(index=False).encode('utf-8'),
                   file_name='toftir_weather_forecast.csv', mime='text/csv')
st.caption('Spread = standard deviation of available model predictions, not a calibrated probability or prediction interval. '
           'The models are not independent; they may share observational inputs and other modelling assumptions. '
           'Temperature is forecast 2 m air temperature, not road temperature. Precipitation is the quantity for the preceding hour. Times shown are local Faroe Islands time. '
           'All displayed times use 24-hour format (HH:MM). Open-Meteo forecast data: https://open-meteo.com/')
