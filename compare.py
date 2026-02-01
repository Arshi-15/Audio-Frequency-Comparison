import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
ref_audio, sr1 = librosa.load("audio1.wav.ogg", sr=None)
pat_audio, sr2 = librosa.load("audio2.wav.ogg", sr=None)
min_len = min(len(ref_audio), len(pat_audio))
ref_audio = ref_audio[:min_len]
pat_audio = pat_audio[:min_len]
time = np.linspace(0, min_len / sr1, min_len)
fft_ref = np.abs(np.fft.fft(ref_audio))
fft_pat = np.abs(np.fft.fft(pat_audio))
fft_ref = fft_ref[:min_len // 2]
fft_pat = fft_pat[:min_len // 2]
similarity_score = 1 - cosine(fft_ref, fft_pat)
print("Similarity Score:", round(similarity_score, 4))
plt.figure(figsize=(10, 5))
plt.plot(time, ref_audio, label="Reference (ref)", color="blue")
plt.plot(time, -pat_audio, label="Pattern (pat)", color="orange")
plt.title("Audio Signal Comparison")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.text(
    0.02, 0.95,
    f"Similarity Score: {similarity_score:.4f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top"
)
plt.tight_layout()
plt.show()