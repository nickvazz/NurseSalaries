import pandas as pd
import numpy as np
import geocoder, os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib import cm
from mpl_toolkits.basemap import Basemap


import folium
from folium.plugins import MarkerCluster
from folium.plugins import BeautifyIcon

from get_api_key import api_key

def load_file():
    print ('loaded')
    if os.path.isfile('fixedNurseSalaryCA.csv'):
        df = pd.read_csv('fixedNurseSalaryCA.csv')
    else:
        df = pd.read_csv('WagesCaRN.csv', skiprows=1)

        df = df.loc[2:,['Hospital','City','New Grad Base Pay','Salary (Estimate)']]
        df.columns = ['Hospital','City','Hourly','Yearly']
        df.iloc[49]['Yearly'] =0
        df = df.reset_index(drop=True)

        def fix_money_string(x):
            if type(x) == str:
                return float(x.replace('$','').replace(',',''))
            else:
                return x

        def get_loc(x):
            g = geocoder.bing(x, key=api_key(), locality='California', maxRows=1)
            return (g.latlng)

        df['Hourly'] = df['Hourly'].apply(fix_money_string)
        df['Yearly'] = df['Yearly'].apply(fix_money_string)
        df['Search'] = df['Hospital'] + ' ' + df['City'] + ' California'
        df['Lat Long'] = df['Search'].apply(get_loc)

        df = pd.concat([df, pd.DataFrame(df['Lat Long'].values.tolist(), columns=['Lat', 'Long'])], axis=1)

        df = df[['Hospital','Hourly','Yearly','Lat','Long']]
        df['Yearly'] = df['Yearly'].fillna(value=df['Hourly']*2080)
        df = df[['Hospital','Yearly','Lat','Long']]

        df['Yearly'] = df['Yearly'].fillna(0)
        df = df[df['Yearly'] != 0]
        df = df[df['Lat'] > 30]
        df = df[df['Long'] < 0]
        df.reset_index(drop=True)
        df.to_csv('fixedNurseSalaryCA.csv', index=False)

    return df


def make_basemap_image(df):
    print (df.head())
    scaler = MinMaxScaler(feature_range=(10,100))
    size = scaler.fit_transform(df['Yearly'].values.reshape(-1,1))

    fig = plt.figure(figsize=(8,8))
    m = Basemap(projection='lcc', resolution='h',
                lat_0=df['Lat'].mean(), lon_0=df['Long'].mean(),
                width=1E6*.75, height=1.2E6*.75)

    m.shadedrelief()
    m.drawcoastlines(color='black')
    m.drawcountries(color='gray')
    m.drawstates(color='black')

    parallels = np.arange(0.,81,2.)
    meridians = np.arange(-360,351.,2.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[False,True,False,False])
    m.drawmeridians(meridians,labels=[False,False,False,True])

    m.scatter(df['Long'].values, df['Lat'].values, 
              latlon=True, c=df['Yearly'].values,
              cmap='coolwarm', alpha=0.75,
              s=size, marker='o')

    plt.savefig('scatterMap.png')
    

def make_html(df):
    print (df.head())
    scaler = MinMaxScaler(feature_range=(0,1))
    size = scaler.fit_transform(df['Yearly'].values.reshape(-1,1))

    cmap = cm.get_cmap('coolwarm')

    m = folium.Map(location=[df['Lat'].mean(), df['Long'].mean()], zoom_start=6.3)

    marker_cluster = MarkerCluster().add_to(m)

    for idx in range(len(df)):
        rgb = cmap(size[idx])[:3]
        color = rgb2hex(rgb[0])

        folium.Marker(
            location=[df['Lat'][idx], df['Long'][idx]],
            popup='''{}<br><center>${:,.2f} per year</center>'''.format(df['Hospital'][idx], df['Yearly'][idx]),
            icon=folium.Icon(color='red', icon='heart',icon_color=color),
        ).add_to(marker_cluster)

    m.save('CaNurseSalaries.html')


def main():
    df = load_file()
    # make_basemap_image(df)
    make_html(df)
    
if __name__ == '__main__':
    main()