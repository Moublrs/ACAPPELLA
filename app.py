import os
import tempfile
import datetime
from pathlib import Path

import torch
import numpy as np
import librosa
import soundfile as sf
import yt_dlp
import gradio as gr
import requests

from model import UNet

# =========================
# Configuration
# =========================
GPU_ID = 7
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(GPU_ID)

SR = 8192
N_FFT = 1024
HOP_LENGTH = 768
FRAME_SIZE = 128
STRIDE_FRAMES = 64

# =========================
# Modèles disponibles
# =========================
MODELS = {
    "musdb": "./models/unet_final.pth",
    "Fine-tuned": "./models/unet_final_fine_tuned.pth"
}

models = {}
for name, path in MODELS.items():
    m = UNet().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    m.eval()
    models[name] = m
    print(f" Modèle chargé: {name} ({path})")

# =========================
# Téléchargement YouTube
# =========================
def download_from_youtube(query):
    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "default_search": "ytsearch1",
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch:{query}", download=True)
        
        if not info:
            return None, None, None, None, None
        
        if "entries" in info:
            if not info["entries"]:
                return None, None, None, None, None
            video_info = info["entries"][0]
        else:
            video_info = info
        
        title = video_info.get("title", "Inconnu")
        thumbnail_url = video_info.get("thumbnail", "")
        channel = video_info.get("channel", "Inconnu")
        duration = video_info.get("duration", 0)
        view_count = video_info.get("view_count", 0)
        video_id = video_info.get("id", "")
        
        mins, secs = divmod(duration, 60)
        print(f"    TROUVÉ: {title}")
        print(f"    Chaîne: {channel} |  {int(mins)}:{int(secs):02d} |  {view_count:,} vues")
        print(f"    https://youtube.com/watch?v={video_id}")
        
        thumbnail_path = None
        if thumbnail_url:
            try:
                resp = requests.get(thumbnail_url, timeout=10)
                if resp.status_code == 200:
                    thumbnail_path = tempfile.mktemp(suffix=".jpg")
                    with open(thumbnail_path, "wb") as f:
                        f.write(resp.content)
            except:
                pass

        mp3_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp3")]
        audio_path = max(mp3_files, key=os.path.getctime) if mp3_files else None
        
        return audio_path, title, thumbnail_path, channel, duration
    
    except Exception as e:
        print(f"    Erreur: {e}")
        return None, None, None, None, None

# =========================
# Séparation vocale U-Net
# =========================
def _separate_from_spectrogram(mix_mag, model):
    """Cœur de la séparation : prend magnitude, retourne mask appliqué"""
    mix_max = mix_mag.max() + 1e-8
    mix_norm = (mix_mag / mix_max)[:-1, :]
    F, T = mix_norm.shape

    voc_norm = np.zeros((F, T), dtype=np.float32)
    weight = np.zeros((F, T), dtype=np.float32)

    with torch.no_grad():
        for t0 in range(0, T - FRAME_SIZE + 1, STRIDE_FRAMES):
            patch = mix_norm[:, t0:t0 + FRAME_SIZE]
            patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
            mask = model(patch_t).squeeze().cpu().numpy()
            voc_norm[:, t0:t0 + FRAME_SIZE] += mask * patch
            weight[:, t0:t0 + FRAME_SIZE] += 1.0

    voc_norm /= np.maximum(weight, 1.0)

    voc_mag = np.vstack([voc_norm * mix_max, np.zeros((1, T))])
    inst_mag = mix_mag - voc_mag
    
    return voc_mag, inst_mag


def separate_vocals_from_audio(audio_path, model_name):
    """Séparation à partir d'un fichier audio"""
    model = models[model_name]
    
    y,_ = librosa.load(audio_path,sr=SR,mono=True)
    stft_mix = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mix_mag = np.abs(stft_mix).astype(np.float32)
    mix_phase = np.angle(stft_mix).astype(np.float32)
    
    voc_mag, inst_mag = _separate_from_spectrogram(mix_mag, model)
    
    voc_audio = librosa.istft(voc_mag * np.exp(1j * mix_phase), n_fft=N_FFT, hop_length=HOP_LENGTH, length=len(y))
    inst_audio = librosa.istft(inst_mag * np.exp(1j * mix_phase), n_fft=N_FFT, hop_length=HOP_LENGTH, length=len(y))

    vocals_path = tempfile.mktemp(suffix="_vocals.wav")
    inst_path = tempfile.mktemp(suffix="_instruments.wav")
    sf.write(vocals_path, voc_audio, SR)
    sf.write(inst_path, inst_audio, SR)

    return vocals_path, inst_path


def separate_vocals_from_npy(mag_file, phase_file, model_name):
    """Séparation à partir de fichiers .npy (magnitude + phase)"""
    model = models[model_name]
    
    mix_mag = np.load(mag_file.name).astype(np.float32)
    mix_phase = np.load(phase_file.name).astype(np.float32)
    
    voc_mag, inst_mag = _separate_from_spectrogram(mix_mag, model)
    
    voc_audio = librosa.istft(voc_mag * np.exp(1j * mix_phase), n_fft=N_FFT, hop_length=HOP_LENGTH)
    inst_audio = librosa.istft(inst_mag * np.exp(1j * mix_phase), n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    vocals_path = tempfile.mktemp(suffix="_vocals.wav")
    inst_path = tempfile.mktemp(suffix="_instruments.wav")
    sf.write(vocals_path, voc_audio, SR)
    sf.write(inst_path, inst_audio, SR)
    
    return vocals_path, inst_path

# =========================
# Pipelines
# =========================
def process_youtube(query, model_name, request: gr.Request):
    if not query.strip():
        return None, None, None, None, "Entre un titre de chanson"

    print(f"\n{'='*60}")
    print(f" REQUÊTE YOUTUBE | {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"    Recherche: {query}")
    print(f"    Modèle: {model_name}")
    print(f"    IP: {request.client.host}")
    print(f"{'='*60}")

    result = download_from_youtube(query)
    
    if not result[0]:
        print("   ❌ Chanson non trouvée\n")
        return None, None, None, None, "Chanson non trouvée"
    
    audio_path, title, thumbnail, channel, duration = result
    
    print(f"    Séparation en cours...")
    vocals_path, inst_path = separate_vocals_from_audio(audio_path, model_name)
    print(f"    Terminé!\n")

    mins, secs = divmod(duration, 60)
    status = f"### {title}\n**Chaîne:** {channel} • **Durée:** {int(mins)}:{int(secs):02d} • **Modèle:** {model_name}"
    
    return thumbnail, audio_path, vocals_path, inst_path, status


def process_npy(mag_file, phase_file, model_name):
    if mag_file is None or phase_file is None:
        return None, None, "Upload magnitude.npy et phase.npy"
    
    print(f"\n{'='*60}")
    print(f" REQUÊTE NPY | {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"    Magnitude: {mag_file.name}")
    print(f"    Phase: {phase_file.name}")
    print(f"    Modèle: {model_name}")
    print(f"{'='*60}")
    
    try:
        print(f"    Séparation en cours...")
        voc_path, inst_path = separate_vocals_from_npy(mag_file, phase_file, model_name)
        print(f"    Terminé!\n")
        return voc_path, inst_path, f"Séparation terminée ✅ (Modèle: {model_name})"
    except Exception as e:
        print(f"    Erreur: {e}\n")
        return None, None, f"Erreur: {e}"

# =========================
# Interface
# =========================
CSS = """
.gradio-container { max-width: 1000px !important; margin: auto; }
.title { text-align: center; margin-bottom: 0; }
.subtitle { text-align: center; color: #666; margin-top: 0; }
"""

with gr.Blocks(title="ACAPPELLA", theme=gr.themes.Soft(), css=CSS) as demo:
    
    gr.Markdown("# ACAPPELLA", elem_classes="title")
    gr.Markdown("*Séparation vocale par intelligence artificielle*", elem_classes="subtitle")
    
    with gr.Tabs():
        # === Tab YouTube ===
        with gr.TabItem(" YouTube"):
            with gr.Row():
                query = gr.Textbox(
                    label="",
                    placeholder="Titre ou artiste (ex: Adele - Hello)",
                    scale=3,
                    container=False
                )
                model_select_yt = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="Fine-tuned",
                    label="Modèle",
                    scale=1
                )
                btn_yt = gr.Button("Extraire", variant="primary", scale=1)
            
            with gr.Row():
                with gr.Column(scale=1):
                    thumbnail = gr.Image(label="", show_label=False, height=200)
                    status_yt = gr.Markdown("")
                
                with gr.Column(scale=2):
                    original = gr.Audio(label="Original")
                    with gr.Row():
                        vocals_yt = gr.Audio(label="Voix")
                        instruments_yt = gr.Audio(label="Instruments")
            
            gr.Examples(
                examples=["Adele - Hello", "Daft Punk - Get Lucky", "Queen - Bohemian Rhapsody"],
                inputs=query,
                label="Essayer"
            )
            
            btn_yt.click(process_youtube, [query, model_select_yt], [thumbnail, original, vocals_yt, instruments_yt, status_yt])
            query.submit(process_youtube, [query, model_select_yt], [thumbnail, original, vocals_yt, instruments_yt, status_yt])
        
        # === Tab NPY ===
        with gr.TabItem(" Spectrogramme (.npy)"):
            gr.Markdown("Upload les fichiers `magnitude.npy` et `phase.npy` du mix")
            
            with gr.Row():
                mag_input = gr.File(label="Magnitude (.npy)", file_types=[".npy"])
                phase_input = gr.File(label="Phase (.npy)", file_types=[".npy"])
                model_select_npy = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="Fine-tuned",
                    label="Modèle"
                )
            
            btn_npy = gr.Button("Séparer", variant="primary")
            status_npy = gr.Markdown("")
            
            with gr.Row():
                vocals_npy = gr.Audio(label="Voix")
                instruments_npy = gr.Audio(label="Instruments")
            
            btn_npy.click(process_npy, [mag_input, phase_input, model_select_npy], [vocals_npy, instruments_npy, status_npy])

# =========================
# Lancement
# =========================
print(f" ACAPPELLA | GPU: {torch.cuda.get_device_name(GPU_ID)} | Device: {DEVICE}")

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)