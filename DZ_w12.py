import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import numpy as np
import math  # Import math module for gamma function
from sklearn.linear_model import LinearRegression

# Configuration de la page
st.set_page_config(page_title="Atlas des Vents Algérien", layout="wide")

# Titre de l'application
st.title("📌 Atlas des Vents - Algérie")

@st.cache_data
def load_data():
    """Load GeoJSON data for wind and administrative boundaries."""
    try:
        gdf = gpd.read_file("dzv.geojson")
        admin = gpd.read_file("all-wilayas (1).geojson")
        return gdf, admin
    except Exception as e:
        st.error(f"Erreur de chargement des données : {e}")
        return None, None

@st.cache_data
def load_csv():
    """Load CSV file with wind data."""
    try:
        df = pd.read_csv("df_10.csv")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier CSV : {e}")
        return None

# Fonction for mathematical calculations

def ml(a, b):
    try:
        if a != 0 and b != 0:
            c = a / math.gamma(1 + 1 / b)
        else:
            c = 0
    except OverflowError:
        c = 0
    return c

def ml(a, b):
    try:
        if (a != 0) and (b != 0):
            c = a / math.gamma(1 + 1 / b)
        else:
            c = 0
    except OverflowError:
        c = 0
    return c
# Vectorisation de la fonction ml
ml_vectorized = np.vectorize(ml)

def estimer_vitesse_saisonniere(modele, latitude, longitude, z):
    """Estimate seasonal wind speed."""
    return modele.predict([[latitude, longitude, z]])[0]

def intro(a, b, c, d, e, f, g):
    """Train the model and estimate wind speeds."""
    aa = np.column_stack((a, b, c))  # Combine independent variables
    bb = np.array(d)  # Dependent variable
    
    modele = LinearRegression()
    modele.fit(aa, bb)  # Train the model
    
    # Estimate seasonal wind speeds
    return estimer_vitesse_saisonniere(modele, e, f, g)

# Load additional data
dfp = pd.read_csv("data1.txt", delimiter="\t").astype(float)
matrice = dfp.to_numpy()
vmat = matrice[:, 4:54:3]  # V speeds
kmat = matrice[:, 3:54:3]  # K values
cmat = matrice[:, 2:54:3]  # C values

### Calculate v3 using mm
##v3 = np.vectorize(mm)(cmat, kmat)  # Apply mm to each element of cmat and kmat
##p = np.round(0.5 * 1.225 * v3, 2)

# Function for interpolating speeds
def vintrpo(X, Y, V, mat, x, y, v):
    dist1 = np.array([intro(X, Y, V, mat[:, i], x, y, v) for i in range(16)]).flatten()
    return np.hstack((dist1, v))  # Proper concatenation

# Example of using vintrpo
def adra(x, y, ve, ke):
    mp = vintrpo(matrice[:, 0], matrice[:, 1], vmat[:, 16], vmat, x, y, ve)
    mp1 = vintrpo(matrice[:, 0], matrice[:, 1], kmat[:, 16], kmat, x, y, ke)
    mp0 = ml_vectorized(mp, mp1)

    A = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre",
         "Novembre", "Décembre", "Automne", "Hiver", "Printemps", "Été", "Annuel"]
    m = {'dist': A, 'c': np.round(mp0, 1), 'k': np.round(mp1, 2), 'v': np.round(mp, 1)}
    return pd.DataFrame(m)

df = load_csv()

# Load wind and administrative data
wind_gdf, admin_gdf = load_data()

if wind_gdf is not None and admin_gdf is not None and df is not None:
    # Create columns for the interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Parameters in sidebar
        with st.sidebar:
            st.header("Paramètres de Visualisation")
            vmin = st.number_input("Valeur Minimale", min_value=0.0, value=0.0)
            vmax = st.number_input("Valeur Maximale", min_value=0.0, value=9.0)
            cmap = st.selectbox(
                "Palette de Couleurs",
                ['hsv', 'viridis', 'plasma', 'magma', 'jet'],
                index=0
            )
            line_width = st.slider("Épaisseur des Frontières", 0.1, 2.0, 0.5)

            st.header("Ajouter un Point")
            lon = st.number_input("Longitude", value=3.0, format="%.6f")
            lat = st.number_input("Latitude", value=36.0, format="%.6f")

        # Create the map
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot wind data
        wind_gdf.plot(
            column='v_min',
            cmap=cmap,
            legend=True,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            legend_kwds={'label': "Vitesse du vent (m/s)", 'orientation': "vertical"}
        )
        
        # Add administrative boundaries
        admin_gdf.boundary.plot(
            ax=ax,
            edgecolor='black',
            linewidth=line_width
        )

        # Add user input point
        if lon and lat:
            user_point = gpd.GeoDataFrame(
                pd.DataFrame({'geometry': [Point(lon, lat)]}),
                crs="EPSG:4326"
            )
            user_point.plot(ax=ax, color='red', markersize=100, marker='o', label='Point saisi')
            ax.legend()

        # Configure the map
        ax.set_title("Distribution des Vitesses de Vent par Wilaya à 10m", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)

    with col2:
        # Find nearest point
        if lon and lat:
            df['distance'] = np.sqrt((df['longitude'] - lon)**2 + (df['latitude'] - lat)**2)
            nearest_point = df.loc[df['distance'].idxmin()]
            df1 = adra(nearest_point['longitude'], nearest_point['latitude'], nearest_point['v'], round(nearest_point['k']))

            # Display the table under the map
            st.dataframe(df1, use_container_width=True)

        # Download data button
        st.download_button(
            label="Télécharger l'atlas éolien",
            data=open("dzv.geojson", "rb").read(),
            file_name="donnees_vent.geojson",
            mime="application/geo+json"
        )

else:
    st.warning("Veuillez vérifier la présence des fichiers GeoJSON et CSV dans le répertoire.")
