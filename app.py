import gradio as gr
import torch
import numpy as np
import librosa
import soundfile as sf
import yt_dlp
import os
import tempfile
import traceback
from pathlib import Path
import requests
from model import UNet
# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 8192
N_FFT = 1024
HOP_LENGTH = 768
FRAME_SIZE = 128
STRIDE_FRAMES = 64

# Importer votre modèle (vous devrez l'adapter)
MODEL_URL = "https://www.dropbox.com/scl/fi/pnzxhaueynzljif7kh86i/unet_final.pth?rlkey=umz3jel4az9wf8j75d0hmx04z&st=2vihy6yj&dl=0"  # À remplacer
MODEL_PATH = "unet_final.pth"

def download_model_if_needed():
    """Télécharge le modèle s'il n'existe pas"""
    if not os.path.exists(MODEL_PATH):
        print("📥 Téléchargement du modèle...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Modèle téléchargé")
    
    # Charger le modèle
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# Au début de votre app
model = download_model_if_needed()

def download_youtube_audio(query):
    """Télécharge audio depuis YouTube"""
    temp_dir = tempfile.mkdtemp()
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(temp_dir, 'song.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'default_search': 'ytsearch1',
        'noplaylist': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch:{query}", download=True)
        
        # Trouver le fichier WAV
        for f in os.listdir(temp_dir):
            if f.endswith('.wav'):
                return os.path.join(temp_dir, f), info.get('title', 'Chanson')
                
    except Exception as e:
        print(f"Erreur téléchargement: {e}")
    
    return None, None

def separate_vocals(audio_path):
    """Sépare les voix (version simplifiée pour début)"""
    if model is None:
        # Mode simulation pour tester
        y, sr = librosa.load(audio_path, sr=SR)
        # Simulation: juste filtrer certaines fréquences
        D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mask = np.ones_like(D)
        # Filtrer les fréquences vocales typiques (80Hz - 1100Hz)
        freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
        vocal_mask = (freqs >= 80) & (freqs <= 1100)
        mask[vocal_mask, :] = 0.8
        mask[~vocal_mask, :] = 0.2
        
        D_vocals = D * mask
        y_vocals = librosa.istft(D_vocals, hop_length=HOP_LENGTH)
        
        temp_out = tempfile.mktemp(suffix='.wav')
        sf.write(temp_out, y_vocals, SR)
        return temp_out
    
    # ICI: votre vraie logique U-Net ira plus tard
    return audio_path  # Temporaire

def process_song(query, progress=gr.Progress()):
    """Fonction principale"""
    if not query.strip():
        return None, None, "❌ Veuillez entrer un titre"
    
    progress(0.2, desc="Recherche YouTube...")
    
    # 1. Téléchargement
    audio_path, title = download_youtube_audio(query)
    if not audio_path:
        return None, None, "❌ Chanson non trouvée"
    
    progress(0.5, desc="Téléchargement terminé")
    
    # 2. Séparation
    progress(0.7, desc="Extraction vocale...")
    vocals_path = separate_vocals(audio_path)
    
    progress(0.9, desc="Finalisation...")
    
    return audio_path, vocals_path, f"✅ '{title}' traité avec succès!"

# Interface Gradio
with gr.Blocks(title="🎵 ACAPPELLA - Extracteur Vocal", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎵 ACAPPELLA - Extracteur Vocal U-Net
    ### *Extrayez les voix des chansons pour la recherche*
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("https://cdn-icons-png.flaticon.com/512/3626/3626048.png", width=100)
            gr.Markdown("**Comment utiliser:**")
            gr.Markdown("""
            1. Entrez un titre de chanson
            2. Cliquez sur 'Extraire'
            3. Écoutez et téléchargez
            """)
            
            query_input = gr.Textbox(
                label="🎤 Titre de la chanson",
                placeholder="Ex: The Weeknd - Blinding Lights",
                lines=2
            )
            
            extract_btn = gr.Button("🚀 Extraire les voix", variant="primary")
            
        with gr.Column(scale=2):
            status = gr.Markdown("**Status:** En attente...")
            
            with gr.Tabs():
                with gr.TabItem("🎧 Original"):
                    original_audio = gr.Audio(label="Chanson originale", type="filepath")
                
                with gr.TabItem("🎤 Voix Extraites"):
                    vocals_audio = gr.Audio(label="Voix isolées", type="filepath")
            
            with gr.Row():
                gr.Markdown("**Télécharger:**")
                download_btn = gr.Button("📥 Télécharger les voix")
    
    # Événements
    extract_btn.click(
        fn=process_song,
        inputs=[query_input],
        outputs=[original_audio, vocals_audio, status]
    )
    
    # Exemples
    gr.Examples(
        examples=[
            ["Adele - Hello"],
            ["Michael Jackson - Billie Jean"],
            ["Queen - Bohemian Rhapsody"],
            ["Ed Sheeran - Shape of You"]
        ],
        inputs=[query_input],
        label="🎵 Exemples rapides"
    )
    
    gr.Markdown("---")
    gr.Markdown("*Pour usage recherche uniquement*")

# Lancement
if __name__ == "__main__":
    demo.launch(debug=True)