import librosa
import numpy as np

class AudioProcessor:
    """
    Classe responsable du traitement des fichiers audio :
    - Chargement d'un fichier audio
    - Normalisation du signal
    - Extraction des coefficients MFCC
    """

    def __init__(self, sample_rate=16000, n_mfcc=13):
        """
        Initialise les paramètres du processeur audio.
        
        :param sample_rate: Taux d'échantillonnage à utiliser pour lire les audios (par défaut 16000 Hz)
        :param n_mfcc: Nombre de coefficients MFCC à extraire (par défaut 13)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def load_audio(self, file_path):
        """
        Charge un fichier audio et retourne le signal et sa fréquence d'échantillonnage.

        :param file_path: Chemin vers le fichier audio (.wav)
        :return: Tuple (signal audio, sample_rate)
        """
        signal, sr = librosa.load(file_path, sr=self.sample_rate)
        return signal, sr

    def normalize(self, signal):
        """
        Normalise le signal audio entre -1 et 1.

        :param signal: Signal audio brut
        :return: Signal normalisé
        """
        return signal / np.max(np.abs(signal))

    def extract_mfcc(self, signal):
        """
        Extrait les coefficients MFCC du signal audio.

        :param signal: Signal audio normalisé
        :return: Tableau numpy des MFCC (shape : (n_mfcc, nombre de frames))
        """
        mfccs = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        return mfccs

    def process(self, file_path):
        """
        Traite complètement un fichier audio :
        Chargement, normalisation et extraction MFCC.

        :param file_path: Chemin vers le fichier audio
        :return: MFCC du fichier audio
        """
        signal, _ = self.load_audio(file_path)
        normalized_signal = self.normalize(signal)
        mfccs = self.extract_mfcc(normalized_signal)
        return mfccs
# Exemple d'utilisation
if __name__ == "__main__":
        # Créer une instance du processeur
    audio_processor = AudioProcessor(sample_rate=16000, n_mfcc=13)

    # Traiter un fichier audio
    mfcc = audio_processor.process("dataset_fon/armandine/armandine_fongbe_corp005_001.wav")

    # Afficher la forme du tableau MFCC
    print(mfcc.shape)

    print(mfcc)  # Affiche les coefficients MFCC  