import os
import pandas as pd

# Chemin de base
base_dir = "dataset_fon"
transcription_file = os.path.join(base_dir, "metadata", "text")

# Lire les transcriptions avec noms de fichiers
with open(transcription_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Préparer les listes pour le DataFrame
audio_paths = []
transcriptions = []
speakers = []

# Lire chaque ligne
for line in lines:
    parts = line.strip().split()
    if len(parts) < 2:
        continue  # ignorer les lignes incomplètes

    filename = parts[0] + ".wav"
    transcription = " ".join(parts[1:])

    # Parcourir tous les dossiers pour chercher ce fichier audio
    found = False
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            audio_path = os.path.join(root, filename)
            speaker = os.path.basename(os.path.dirname(audio_path))

            audio_paths.append(audio_path)
            transcriptions.append(transcription)
            speakers.append(speaker)
            found = True
            break

    if not found:
        print(f"Fichier non trouvé : {filename}")

# Créer le DataFrame
df = pd.DataFrame({
    "audio_path": audio_paths,
    "transcription": transcriptions,
    "speaker": speakers
})

# Sauvegarder le metadata final
df.to_csv(os.path.join(base_dir, "metadata", "metadata.csv"), index=False, encoding="utf-8")

print("metadata.csv généré avec succès.")
