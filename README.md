# ACAPPELLA

Application web de séparation vocale par intelligence artificielle.

## Fonctionnalités

- Recherche et téléchargement automatique depuis YouTube
- Séparation vocals/instruments en temps réel
- Comparaison de deux modèles (Base vs Fine-tuned)
- Interface pour spectrogrammes numpy (.npy)

## Déploiement

L'application est hébergée sur un serveur GPU distant. Pour y accéder :

1. Je lance l'application sur le serveur :
```bash
python app.py
```

2. J'expose le port via zrok :
```bash
zrok share public localhost:7860
```

3. Un lien public est généré et partagé pour accès externe.

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Interface | Gradio |
| Modèle | U-Net (PyTorch) |
| Téléchargement | yt-dlp |
| Audio | librosa, soundfile |
| Tunneling | zrok / Gradio share |

## Paramètres d'Inférence

| Paramètre | Valeur |
|-----------|--------|
| Sample rate | 8192 Hz |
| FFT size | 1024 |
| Hop length | 768 |
| Frame size | 128 |
| Stride | 64 (50% overlap) |

## Défis Rencontrés

**yt-dlp** : Gestion des métadonnées variables, vidéos géo-bloquées, extraction audio via FFmpeg.

**Tunneling** : Plusieurs solutions testées (ngrok payant, bore down, cloudflared bloqué). Solution retenue : zrok ou Gradio share intégré.

## Modèles Disponibles

- **Base** : Entraîné sur MUSDB18 (100 pistes)
- **Fine-tuned** : Fine-tuné sur données YouTube + Demucs (428 pistes)

## Auteur

Mouad
