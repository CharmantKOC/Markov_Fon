import numpy as np
from Audioprocess import AudioProcessor
from encoderMfcc import MFCCEncoder
from Markovmodelfon import CustomHMM

class SpeechRecognizer:
    def __init__(self, audio_processor, mfcc_encoder, hmm_model, state_to_text):
        """
        Initialise le SpeechRecognizer.
        
        :param audio_processor: instance de la classe AudioProcessor.
        :param mfcc_encoder: instance de la classe MFCCEncoder.
        :param hmm_model: instance de la classe HMMModel.
        :param state_to_text: dictionnaire associant les états à des phonèmes ou mots.
        """
        self.audio_processor = audio_processor
        self.mfcc_encoder = mfcc_encoder
        self.hmm_model = hmm_model
        self.state_to_text = state_to_text

    def transcribe(self, audio_path):
        """
        Transcrit un fichier audio en texte.
        
        :param audio_path: chemin du fichier audio à transcrire.
        :return: transcription textuelle.
        """
        # 1️⃣ Extraction des MFCC
        mfcc_features = self.audio_processor.extract_mfcc(audio_path)
        
        # 2️⃣ Encodage des MFCC en observations discrètes
        encoded_obs = self.mfcc_encoder.encode(mfcc_features)
        
        # 3️⃣ Application de Viterbi pour obtenir la séquence d'états
        states_seq, prob = self.hmm_model.viterbi(encoded_obs)
        
        # 4️⃣ Traduction des états en texte
        transcription = self._states_to_text(states_seq)
        
        return transcription, prob

    def _states_to_text(self, states_seq):
        """
        Convertit une séquence d'états en texte.
        
        :param states_seq: liste ou tableau des états.
        :return: string transcription.
        """
        text = []
        for state in states_seq:
            text.append(self.state_to_text.get(state, ""))  # On ignore les états non reconnus
        return ' '.join(text)


recognizer = SpeechRecognizer(AudioProcessor, MFCCEncoder,CustomHMM, state_to_text)
transcription, prob = recognizer.transcribe("data/speaker1/audio1.wav")

print("Transcription :", transcription)
print("Probabilité :", prob)
