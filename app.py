from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
os.makedirs('uploads', exist_ok=True)

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

GENRE_INFO = {
    'blues': {'emoji': '🎸', 'color': '#1a3a5c', 'desc': 'Soulful melodies with deep emotional expression originating from African American communities.'},
    'classical': {'emoji': '🎻', 'color': '#2c1810', 'desc': 'Structured compositions featuring orchestral instruments with rich harmonic complexity.'},
    'country': {'emoji': '🤠', 'color': '#3d2b1f', 'desc': 'Storytelling through music with guitar-driven sounds rooted in American folk traditions.'},
    'disco': {'emoji': '🕺', 'color': '#4a0e4e', 'desc': 'High-energy dance music with strong basslines and orchestral elements from the 70s.'},
    'hiphop': {'emoji': '🎤', 'color': '#1a1a2e', 'desc': 'Rhythm-heavy genre featuring beats, sampling, and vocal performance with cultural roots.'},
    'jazz': {'emoji': '🎺', 'color': '#1c2833', 'desc': 'Improvisational music with complex harmonies, swing rhythms and expressive freedom.'},
    'metal': {'emoji': '🤘', 'color': '#0d0d0d', 'desc': 'High-intensity genre with distorted guitars, powerful drumming and aggressive energy.'},
    'pop': {'emoji': '🎵', 'color': '#1a0533', 'desc': 'Catchy, commercially crafted music designed for wide appeal with polished production.'},
    'reggae': {'emoji': '🇯🇲', 'color': '#0a2e0a', 'desc': 'Laid-back rhythms from Jamaica with offbeat guitar strums and conscious lyrics.'},
    'rock': {'emoji': '🎸', 'color': '#2d0a0a', 'desc': 'Guitar-driven music with strong rhythms ranging from soft rock to hard rock variants.'}
}

def extract_features(file_path):
    """Extract audio features for genre classification."""
    y, sr = librosa.load(file_path, duration=30)
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing = librosa.feature.zero_crossing_rate(y)[0]
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    features = {
        'mfcc_mean': mfccs_mean.tolist(),
        'mfcc_std': mfccs_std.tolist(),
        'chroma_mean': chroma_mean.tolist(),
        'spectral_centroid_mean': float(np.mean(spectral_centroids)),
        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
        'zero_crossing_mean': float(np.mean(zero_crossing)),
        'tempo': float(tempo)
    }
    return features, y, sr

def generate_spectrogram(y, sr):
    """Generate mel spectrogram image as base64."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    
    ax.set_xlabel('Time (s)', color='#888', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', color='#888', fontsize=10)
    ax.tick_params(colors='#666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    
    plt.colorbar(img, ax=ax, format='%+2.0f dB').ax.yaxis.set_tick_params(color='#666')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a0f')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_waveform(y, sr):
    """Generate waveform image as base64."""
    fig, ax = plt.subplots(figsize=(10, 2), facecolor='#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    
    times = np.linspace(0, len(y)/sr, len(y))
    ax.plot(times, y, color='#00d4ff', linewidth=0.5, alpha=0.8)
    ax.fill_between(times, y, alpha=0.3, color='#00d4ff')
    
    ax.set_xlabel('Time (s)', color='#888', fontsize=9)
    ax.set_yticks([])
    ax.tick_params(colors='#666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a0f')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def mock_predict(features):
    """
    Mock prediction - returns realistic scores based on audio features.
    In production, replace this with your trained CNN/LSTM model.
    """
    # Use features to create somewhat realistic scores
    tempo = features['tempo']
    zcr = features['zero_crossing_mean']
    centroid = features['spectral_centroid_mean']
    
    # Base scores
    scores = {genre: random.uniform(0.02, 0.15) for genre in GENRES}
    
    # Feature-based boosting (simulates model logic)
    if tempo > 140:
        scores['metal'] += 0.3
        scores['rock'] += 0.2
        scores['disco'] += 0.15
    elif tempo > 120:
        scores['pop'] += 0.25
        scores['hiphop'] += 0.2
        scores['rock'] += 0.1
    elif tempo > 100:
        scores['country'] += 0.2
        scores['reggae'] += 0.15
        scores['blues'] += 0.1
    else:
        scores['classical'] += 0.3
        scores['jazz'] += 0.25
        scores['blues'] += 0.15
    
    if zcr > 0.1:
        scores['metal'] += 0.15
        scores['rock'] += 0.1
    
    if centroid > 3000:
        scores['pop'] += 0.1
        scores['disco'] += 0.1
    elif centroid < 1500:
        scores['classical'] += 0.1
        scores['jazz'] += 0.1
    
    # Normalize
    total = sum(scores.values())
    scores = {k: v/total for k, v in scores.items()}
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    allowed = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': f'File type .{ext} not supported. Use: mp3, wav, ogg, flac, m4a'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        features, y, sr = extract_features(filepath)
        predictions = mock_predict(features)
        spectrogram = generate_spectrogram(y, sr)
        waveform = generate_waveform(y, sr)
        
        top_genre = predictions[0][0]
        genre_data = GENRE_INFO.get(top_genre, {})
        
        result = {
            'top_genre': top_genre,
            'emoji': genre_data.get('emoji', '🎵'),
            'color': genre_data.get('color', '#333'),
            'description': genre_data.get('desc', ''),
            'confidence': round(predictions[0][1] * 100, 1),
            'all_predictions': [
                {
                    'genre': g,
                    'score': round(s * 100, 1),
                    'emoji': GENRE_INFO.get(g, {}).get('emoji', '🎵')
                }
                for g, s in predictions
            ],
            'features': {
                'tempo': round(features['tempo'], 1),
                'spectral_centroid': round(features['spectral_centroid_mean'], 1),
                'zero_crossing_rate': round(features['zero_crossing_mean'], 4),
                'spectral_bandwidth': round(features['spectral_bandwidth_mean'], 1),
            },
            'spectrogram': spectrogram,
            'waveform': waveform,
            'duration': round(len(y) / sr, 1),
            'sample_rate': sr
        }
        
        os.remove(filepath)
        return jsonify(result)
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
