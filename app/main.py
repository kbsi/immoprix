import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

import pandasql as psql

import joblib
import os


def load_data(filepath):
    # Load the data
    dvf = pd.read_csv(filepath, sep=',', low_memory=False)

    # Filter dvf on nature_mutation = Vente, Adjudication et Echange
    # Filter on type_local house, apartment, dependency
    # Remove multiple sales (e.g., entire buildings, subdivisions), all id_mutation that have multiple code_type_local = 1 or 2
    query = """
    select
        id_mutation,
        date_mutation,
        valeur_fonciere,
        code_commune,
        longitude,
        latitude,
        code_type_local,
        nombre_pieces_principales,
        code_nature_culture,
        nature_culture,
        surface_reelle_bati,
        surface_terrain
    from dvf
    where id_mutation in (
        select id_mutation
        from dvf
        where code_type_local in (1, 2)
        group by id_mutation
        having count(distinct code_type_local) = 1
    )
    and nature_mutation in ('Vente', 'Adjudication', 'Echange')
    and code_type_local in (1, 2, 3)
    """
    dvf_train = psql.sqldf(query, locals())

    # Maisons et appartements
    query = """
    select distinct
        id_mutation,
        date_mutation,
        valeur_fonciere,
        code_commune,
        longitude,
        latitude,
        code_type_local,
        nombre_pieces_principales,
        nature_culture,
        surface_reelle_bati,
        surface_terrain
    from dvf 
    where code_type_local in (1,2) and code_nature_culture in ('S', '')
    """
    dvf_train = psql.sqldf(query)

    # on extrait les differentes superficies disponibles
    # T : terres
    # J : jardin
    # AG: terrains d'agréments
    # autre

    query = """
    select dvf.id_mutation, sum(dvf_train.surface_terrain) as surface_terres
    from dvf 
    right join dvf_train on dvf.id_mutation = dvf_train.id_mutation
    where 
        dvf.code_type_local = 3
        and dvf.code_nature_culture = 'T'
    group by dvf.id_mutation
    """
    dvf_train_terres = psql.sqldf(query)

    query = """
    select dvf.id_mutation, sum(dvf_train.surface_terrain) as surface_jardins
    from dvf 
    right join dvf_train on dvf.id_mutation = dvf_train.id_mutation
    where 
        dvf.code_type_local = 3
        and dvf.code_nature_culture = 'J'
    group by dvf.id_mutation
    """
    dvf_train_jardin = psql.sqldf(query)

    query = """
    select dvf.id_mutation, sum(dvf_train.surface_terrain) as surface_terrain_agrement
    from dvf 
    right join dvf_train on dvf.id_mutation = dvf_train.id_mutation
    where 
        dvf.code_type_local = 3
        and dvf.code_nature_culture = 'AG'
    group by dvf.id_mutation
    """
    dvf_train_agrement = psql.sqldf(query)

    query = """
    select dvf.id_mutation, sum(dvf_train.surface_terrain) as surface_terrain_autre
    from dvf 
    right join dvf_train on dvf.id_mutation = dvf_train.id_mutation
    where 
        dvf.code_type_local = 3
        and dvf.code_nature_culture not in ('T', 'J', 'AG')
    group by dvf.id_mutation
    """
    dvf_train_autre = psql.sqldf(query)

    query = """
    select distinct
        dvf_train.id_mutation,

        dvf_train.code_type_local,

        dvf_train.date_mutation,
        dvf_train.valeur_fonciere,
        dvf_train.code_commune,
        dvf_train.longitude,
        dvf_train.latitude,
        dvf_train.nombre_pieces_principales,
        dvf_train.surface_reelle_bati,
        dvf_train.surface_terrain,

        dvf_train_terres.surface_terres,
        dvf_train_jardin.surface_jardins,
        dvf_train_agrement.surface_terrain_agrement,
        dvf_train_autre.surface_terrain_autre
    from dvf_train 
    left join dvf_train_terres on dvf_train.id_mutation = dvf_train_terres.id_mutation
    left join dvf_train_jardin on dvf_train.id_mutation = dvf_train_jardin.id_mutation
    left join dvf_train_agrement on dvf_train.id_mutation = dvf_train_agrement.id_mutation
    left join dvf_train_autre on dvf_train.id_mutation = dvf_train_autre.id_mutation
    """
    dvf_train = psql.sqldf(query)
    return dvf_train


def load_and_merge_insee_data(filepath, dvf_train):
    # on load les données de l'insee
    dvf_insee = pd.read_csv(filepath, sep=';', low_memory=False)

    dvf_insee = dvf_insee.rename(columns={'Code': 'code_commune'})
    dvf_insee = dvf_insee.rename(columns={'Libellé': 'libelle'})
    dvf_insee = dvf_insee.rename(
        columns={'Médiane du niveau de vie 2021': 'niveau_vie'})
    dvf_insee = dvf_insee.rename(
        columns={"Taux d'activité par tranche d'âge 2021": 'taux_activite'})
    dvf_insee = dvf_insee.rename(columns={
                                 'Part des actifs occupés de 15 ans ou plus  les transports en commun 2021': 'transport_commun'})
    dvf_insee = dvf_insee.rename(
        columns={'Population municipale 2021': 'population'})
    dvf_insee = dvf_insee.rename(columns={
                                 "Part des diplômés d'un BAC+5 ou plus dans la pop. non scolarisée de 15 ans ou + 2021": 'diplome'})
    dvf_insee = dvf_insee.rename(
        columns={"École maternelle, primaire, élémentaire (en nombre) 2023": 'ecole'})
    dvf_insee = dvf_insee.rename(
        columns={"Médecin généraliste (en nombre) 2023": 'medecin'})

    # on met 0 pour les valeurs NA
    dvf_insee.fillna(0, inplace=True)

    # on merge avec les données de l'insee
    dvf_train = psql.sqldf("""
            select
                dvf_train.*,
                dvf_insee.niveau_vie,
                dvf_insee.taux_activite,
                dvf_insee.transport_commun,
                dvf_insee.population,
                dvf_insee.diplome,
                dvf_insee.ecole,
                dvf_insee.medecin
            from dvf_train
            inner join dvf_insee on dvf_train.code_commune = dvf_insee.code_commune
        """, locals())

    return dvf_train


def data_preprocessing(dvf_train):
    # Convertir les colonnes
    convert_column_to_float(dvf_train, 'surface_reelle_bati')
    convert_column_to_float(dvf_train, 'valeur_fonciere')
    convert_column_to_float(dvf_train, 'surface_terrain')
    convert_column_to_float(dvf_train, 'longitude')
    convert_column_to_float(dvf_train, 'latitude')
    convert_column_to_int(dvf_train, 'nombre_pieces_principales')
    convert_column_to_int(dvf_train, 'code_commune')
    convert_column_to_date(dvf_train, 'date_mutation')
    convert_column_to_int(dvf_train, 'code_type_local')

    # Extraire des informations temporelles
    dvf_train['annee_mutation'] = dvf_train['date_mutation'].dt.year
    dvf_train['mois_mutation'] = dvf_train['date_mutation'].dt.month

    # on enlève date_mutation
    dvf_train.drop(['date_mutation'], axis=1, inplace=True)

    # on enlève les données manquantes, valeur_fonciere
    dvf_train = dvf_train.dropna(subset=['valeur_fonciere'])

    # on met une valeur moyenne pour longitude et latitude manquantes
    dvf_train.loc[:, 'longitude'] = dvf_train['longitude'].fillna(
        dvf_train['longitude'].mean())
    dvf_train.loc[:, 'latitude'] = dvf_train['latitude'].fillna(
        dvf_train['latitude'].mean())

    # on peut donc regrouper les maisons avec plus de 10 pièces
    dvf_train.loc[dvf_train['nombre_pieces_principales']
                  > 10, 'nombre_pieces_principales'] = 10
    dvf_train.loc[dvf_train['nombre_pieces_principales']
                  == 0, 'nombre_pieces_principales'] = 1

    # on ne garde que les lignes où la valeur foncière est inférieure à 1 300 000€ et supérieure à 1000€
    dvf_train = dvf_train[dvf_train['valeur_fonciere'] > 1000]
    dvf_train = dvf_train[dvf_train['valeur_fonciere'] < 1300000]

    dvf_train = log_transform(dvf_train, 'surface_reelle_bati')
    dvf_train.loc[dvf_train['surface_terrain']
                  > 6000, 'surface_terrain'] = 6000
    dvf_train = log_transform(dvf_train, 'surface_terrain')

    return dvf_train


def log_transform(data, column):
    # Log transform the column
    data[column] = np.log1p(data[column])
    return data


def convert_column_to_date(dvf_train, column):
    dvf_train[column] = pd.to_datetime(dvf_train[column], format='%Y-%m-%d')


def convert_column_to_int(dvf_train, column):
    dvf_train[column] = dvf_train[column].str.replace(
        ',', '.').astype(int)


def convert_column_to_float(dvf_train, column):
    dvf_train[column] = dvf_train[column].str.replace(
        ',', '.').astype(float)


def perform_kmeans_clustering(data, cols, n_clusters=15, save_model=False, model_path=None):
    # Normalize the data
    scaler = StandardScaler()
    X = data[cols].dropna()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    data.loc[X.index, 'cluster'] = kmeans.labels_

    # Save the clustering model
    if save_model and model_path:
        joblib.dump(kmeans, model_path)

    return data


def perform_agglomerative_clustering(data, cols, n_clusters=15, save_model=False, model_path=None):
    # Normalize the data
    scaler = StandardScaler()
    X = data[cols].dropna()
    X_scaled = scaler.fit_transform(X)

    # Apply Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(
        n_clusters=n_clusters).fit(X_scaled)
    data.loc[X.index, 'cluster'] = agg_clustering.labels_

    # Save the clustering model
    if save_model and model_path:
        joblib.dump(agg_clustering, model_path)

    return data


def train_random_forest(dvf_train):
    dvf_train_final = dvf_train[[
        'valeur_fonciere',

        'code_type_local',
        'code_commune',
        'nombre_pieces_principales',
        'surface_reelle_bati',
        'surface_terrain',
        'surface_jardins',
        'surface_terres',
        'surface_terrain_agrement',
        'surface_terrain_autre',
        'annee_mutation',
        'mois_mutation',
        # insee
        'diplome',
        'niveau_vie',
        'taux_activite',
        'medecin',
        'ecole',
        'transport_commun',
        # compute
        'cluster',
    ]]

    # on sépare les données en données d'entrainement et données de test
    X_train, X_test, y_train, y_test = train_test_split(dvf_train_final.drop(
        ['valeur_fonciere'], axis=1), dvf_train_final['valeur_fonciere'], test_size=0.20, random_state=42)

    rf = RandomForestRegressor(n_estimators=950, random_state=42)

    rf.fit(X_train, y_train)

    return rf


def train_extra_tree_regressor(dvf_train):
    # Prepare the final training data
    dvf_train_final = dvf_train[[
        'valeur_fonciere',

        'code_type_local',
        'code_commune',
        'nombre_pieces_principales',
        'surface_reelle_bati',
        'surface_terrain',
        'surface_jardins',
        'surface_terres',
        'surface_terrain_agrement',
        'surface_terrain_autre',
        'annee_mutation',
        'mois_mutation',
        # insee
        'diplome',
        'niveau_vie',
        'taux_activite',
        'medecin',
        'ecole',
        'transport_commun',
        # compute
        'cluster',
    ]]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(dvf_train_final.drop(
        ['valeur_fonciere'], axis=1), dvf_train_final['valeur_fonciere'], test_size=0.20, random_state=42)

    rf = ExtraTreesRegressor(n_estimators=1000, random_state=42)
    rf.fit(X_train, y_train)

    return rf


def get_model(dvf_train, cluster_model_path, model_path, retrain=False):
    # Load the existing model if it exists
    if not retrain and model_path and os.path.exists(model_path):
        rf = joblib.load(model_path)
        return rf

    # Apply clustering
    perform_kmeans_clustering(dvf_train, [
        'longitude', 'latitude'], n_clusters=400, save_model=True, model_path=cluster_model_path)

    rf = train_extra_tree_regressor(dvf_train)

    # Save the model
    if model_path:
        joblib.dump(rf, model_path)

    return rf


def predict_value(model, to_predict, filepath_cluster_model):

    # Prétraiter les nouvelles données de la même manière que les données d'entraînement
    to_predict = load_and_merge_insee_data('./data/data-insee.csv', to_predict)
    to_predict = log_transform(to_predict, 'surface_reelle_bati')
    to_predict = log_transform(to_predict, 'surface_terrain')

    # Load the clustering model
    clustering_model = joblib.load(filepath_cluster_model)

    # Normalize the data
    scaler = StandardScaler()
    X = to_predict[['longitude', 'latitude']].dropna()
    X_scaled = scaler.fit_transform(X)

    # Apply clustering to the new data
    to_predict['cluster'] = clustering_model.predict(X_scaled)

    convert_column_to_date(to_predict, 'date_mutation')
    to_predict['annee_mutation'] = to_predict['date_mutation'].dt.year
    to_predict['mois_mutation'] = to_predict['date_mutation'].dt.month

    to_predict = to_predict[[
        'code_type_local',
        'code_commune',
        'nombre_pieces_principales',
        'surface_reelle_bati',
        'surface_terrain',
        'surface_jardins',
        'surface_terres',
        'surface_terrain_agrement',
        'surface_terrain_autre',
        'annee_mutation',
        'mois_mutation',

        # insee
        'diplome',
        'niveau_vie',
        'taux_activite',
        'medecin',
        'ecole',
        'transport_commun',
        # compute
        'cluster',
    ]]

    # Make predictions
    predictions = model.predict(to_predict)
    return predictions


def main():
    # Load and preprocess the data
    filepath = './data/dvf.csv'
    filepath_insee = './data/data-insee.csv'
    filepath_model = './model/model.pkl'
    filepath_cluster_model = './model/clustering_model.pkl'

    dvf_train = load_data(filepath)
    dvf_train = load_and_merge_insee_data(filepath_insee, dvf_train)
    dvf_train = data_preprocessing(dvf_train)

    model = get_model(dvf_train, filepath_cluster_model,
                      filepath_model)

    # Example of new data for prediction
    new_data = pd.DataFrame({
        'code_type_local': [2, 1, 1, 2],  # 1= house, 2= apartment
        'code_commune': [81140, 81004, 81105, 81105],
        'surface_reelle_bati': [50, 320, 110, 50],
        'nombre_pieces_principales': [3, 9, 5, 3],
        'longitude': [1.813888, 2.1472500, 1.997835, 1.997835],
        'latitude': [43.698897, 43.9258610, 43.765448, 43.765448],
        'date_mutation': ['2025-01-15', '2025-01-15', '2025-01-15', '2025-01-15'],
        'surface_terrain': [0, 400, 200, 0],
        'surface_jardins': [100, 30, 0, 100],
        'surface_terres': [0, 0, 0, 0],
        'surface_terrain_agrement': [0, 0, 0, 0],
        'surface_terrain_autre': [0, 0, 0, 0]
    })

    new_data['prediction'] = predict_value(
        model, new_data, filepath_cluster_model)

    # on affiche new_data sous forme de tableau avec en titre de colonnes : le code commune, la surface réelle bâtie, la surface terrain, le nombre de pièces principales, la valeur foncière prédite
    print(new_data[['code_type_local', 'code_commune', 'surface_reelle_bati',
                    'surface_terrain', 'nombre_pieces_principales', 'prediction']])


if __name__ == "__main__":
    main()
