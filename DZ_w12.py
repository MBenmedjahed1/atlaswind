import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression

# Configuration de la page
st.set_page_config(page_title="Atlas des Vents Algérien", layout="wide")

# Titre de l'application
st.title("📌 Atlas des Vents - Algérie")

# URLs des données
WINDS_DATA_URL = "https://www.dropbox.com/scl/fi/v90aqeqln90kszoduq7kl/dzv.geojson?rlkey=vje6i0mj9gs6hr4gz3j1k68l0&st=kk9659vr&raw=1"
ADMIN_DATA_URL = "https://www.dropbox.com/scl/fi/p04cl3n46j5j0criwrsi2/all-wilayas-1.geojson?rlkey=5a3mlk10uyzwnro4bd1simx73&st=92542j4h&raw=1"
CSV_DATA_URL = "https://www.dropbox.com/scl/fi/wvim31lbdoqaqh6ddwhe8/df_10.csv?rlkey=a70g34p22rnp8emtgf276qngx&st=wax4umm4&raw=1"

@st.cache_data
def load_data():
    """Charge les données géospatiales"""
    try:
        wind_gdf = gpd.read_file(WINDS_DATA_URL)
        admin_gdf = gpd.read_file(ADMIN_DATA_URL)
        return wind_gdf, admin_gdf
    except Exception as e:
        st.error(f"Erreur de chargement des données géospatiales : {str(e)}")
        return None, None

@st.cache_data
def load_csv():
    """Charge les données CSV"""
    try:
        df = pd.read_csv(CSV_DATA_URL)
        return df
    except Exception as e:
        st.error(f"Erreur de chargement du CSV : {str(e)}")
        return None

def ml(a, b):
    """Calcule la vitesse caractéristique selon la distribution de Weibull"""
    try:
        if a > 0 and b > 0:
            return a / math.gamma(1 + 1/b)
        return 0.0
    except (OverflowError, ValueError):
        return 0.0

def train_model(features, target):
    """Entraîne un modèle de régression linéaire"""
    model = LinearRegression()
    model.fit(features, target)
    return model

def estimate_wind_speed(model, lat, lon, elev):
    """Estime la vitesse du vent avec le modèle entraîné"""
    return max(0.0, model.predict([[lat, lon, elev]])[0])

# Chargement des données
wind_gdf, admin_gdf = load_data()
df = load_csv()

if wind_gdf is not None and admin_gdf is not None and df is not None:
    # Configuration de l'interface
    col1, col2 = st.columns([3, 1])

    with col1:
        # Paramètres dans la sidebar
        with st.sidebar:
            st.header("Paramètres de Visualisation")
            vmin = st.number_input("Vitesse minimale (m/s)", min_value=0.0, value=3.0, step=0.5)
            vmax = st.number_input("Vitesse maximale (m/s)", min_value=0.0, value=9.0, step=0.5)
            cmap = st.selectbox(
                "Palette de couleurs",
                ['viridis', 'plasma', 'magma', 'cividis', 'jet'],
                index=0
            )
            
            st.header("Coordonnées Personnalisées")
            lon = st.number_input("Longitude", value=3.0, format="%.6f")
            lat = st.number_input("Latitude", value=36.0, format="%.6f")

        # Création de la carte
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot des données vent
        wind_gdf.plot(
            column='v_min',
            cmap=cmap,
            legend=True,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            legend_kwds={'label': "Vitesse du vent (m/s)", 'shrink': 0.6}
        )
        
        # Frontières administratives
        admin_gdf.boundary.plot(
            ax=ax,
            edgecolor='black',
            linewidth=0.5
        )

        # Marqueur personnalisé
        if lon and lat:
            gpd.GeoDataFrame(
                geometry=[Point(lon, lat)],
                crs="EPSG:4326"
            ).plot(ax=ax, color='red', markersize=80, marker='*')

        ax.set_title("Distribution des Vitesses de Vent à 10m d'altitude", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        st.pyplot(fig)
        plt.close()

    with col2:
        # Calculs dynamiques
        if lat and lon:
            # Trouver le point le plus proche
            df['distance'] = np.hypot(df['latitude']-lat, df['longitude']-lon)
            nearest = df.loc[df['distance'].idxmin()]
            
            # Entraînement du modèle
            features = df[['latitude', 'longitude', 'elevation']]
            target = df['v']
            model = train_model(features, target)
            
            # Estimation
            estimated_v = estimate_wind_speed(model, lat, lon, nearest['elevation'])
            
            # Affichage des résultats
            st.subheader("Estimation du Vent")
            st.metric(label="Vitesse estimée", value=f"{estimated_v:.1f} m/s")
            st.write(f"**Station la plus proche:** {nearest['name']}")
            st.write(f"**Distance:** {nearest['distance']:.2f} degrés")
            
        # Téléchargement des données
        st.download_button(
            label="📥 Télécharger les données",
            data=wind_gdf.to_json().encode(),
            file_name="atlas_vent.geojson",
            mime="application/geo+json"
        )

else:
    st.warning("Impossible de charger les données nécessaires. Vérifiez la connexion internet.")
