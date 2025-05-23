import numpy as np
from sklearn.cluster import KMeans

class MFCCEncoder:
    """
    Quantifie des vecteurs MFCC en symboles discrets à l'aide de KMeans.
    """

    def __init__(self, n_symbols):
        """
        Initialise le quantifieur.

        :param n_symbols: nombre de symboles (clusters) à créer
        """
        self.n_symbols = n_symbols
        self.kmeans = KMeans(n_clusters=n_symbols, random_state=42)

    def fit(self, mfcc_features):
        """
        Entraîne le KMeans sur les MFCC extraits de l'ensemble du dataset.

        :param mfcc_features: tableau numpy de MFCC de dimension (n_frames, n_coeffs)
        """
        self.kmeans.fit(mfcc_features)

    def encode(self, mfcc_features):
        """
        Transforme les MFCC en séquence d'indices de clusters (symboles).

        :param mfcc_features: tableau numpy de MFCC à encoder
        :return: liste d'indices (symboles)
        """
        return self.kmeans.predict(mfcc_features)