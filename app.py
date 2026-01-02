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
MODEL_URL = "https://www.dropbox.com/scl/fi/pnzxhaueynzljif7kh86i/unet_final.pth?rlkey=umz3jel4az9wf8j75d0hmx04z&st=2vihy6yj&dl=1"
MODEL_PATH = "unet_final.pth"

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
        print("📥 Téléchargement du modèle...")

        with requests.get(MODEL_URL, stream=True, allow_redirects=True, timeout=120) as r:
            r.raise_for_status()

            ct = (r.headers.get("Content-Type") or "").lower()
            if "text/html" in ct:
                raise RuntimeError(
                    f"Dropbox a renvoyé du HTML (Content-Type={ct}). "
                    f"Assure-toi d'avoir dl=1 dans l'URL."
                )

            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        # petit check anti-HTML au cas où
        with open(MODEL_PATH, "rb") as f:
            head = f.read(32)
        if head.startswith(b"<!DOCTYPE html") or head.startswith(b"<html") or head.startswith(b"<"):
            raise RuntimeError("Le fichier téléchargé ressemble à une page HTML, pas à un checkpoint PyTorch.")

        print("✅ Modèle téléchargé")

    model = UNet().to(DEVICE)

    # 1) essaie weights_only=True (safe)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except Exception as e:
        raise RuntimeError(
            "Chargement safe (weights_only=True) impossible. "
            "Si tu es 100% sûr de la source, tu peux charger en weights_only=False."
        ) from e

    # 2) gère les 2 formats courants
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model

model = download_model_if_needed()

def download_youtube_audio(query):
    """Télécharge audio depuis YouTube avec fallback"""
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
        # AJOUTEZ CES LIGNES POUR CONTOURNER LES BLOCAGES :
        'socket_timeout': 30,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'tv_embedded', 'web'],
                'skip': ['dash', 'hls']
            }
        },
        # Essayer différents user agents
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        },
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"🔍 Recherche YouTube: {query}")
            info = ydl.extract_info(f"ytsearch:{query}", download=True)
        
        # Trouver le fichier WAV
        for f in os.listdir(temp_dir):
            if f.endswith('.wav'):
                print(f"✅ Audio téléchargé: {f}")
                return os.path.join(temp_dir, f), info.get('title', 'Chanson')
                
    except Exception as e:
        print(f"❌ Erreur téléchargement YouTube: {e}")
        print("🔄 Tentative avec mode démo...")
        
        # FALLBACK: Créer un fichier audio de démo
        return create_demo_audio(query, temp_dir)
    
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