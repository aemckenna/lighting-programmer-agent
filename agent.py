import tkinter
from tkinter import filedialog
import torchaudio
from transformers import pipeline
from pydub import AudioSegment
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal
import soundfile as sf
import torch

# --- Select and Load Audio ---
def select_audio_file():
    root = tkinter.Tk()
    root.withdraw()
    song_path = filedialog.askopenfilename()
    if song_path:
        print("Selected song file:", song_path)
        return song_path
    else:
        print("No song file selected")
        exit()

# --- Convert to WAV ---
def convert_to_wav(song_path, output_path=None):
    if song_path.lower().endswith(".wav"):
        return song_path
    if output_path is None:
        base = os.path.splitext(song_path)[0]
        output_path = base + ".wav"
    audio = AudioSegment.from_file(song_path)
    audio.export(output_path, format="wav")
    print(f"Converted '{song_path}' to '{output_path}'.")
    return output_path

# --- Genre classification ---
def classify_genre(audio_path):
    pipe = pipeline("audio-classification", model="dima806/music_genres_classification")
    return pipe(audio_path)

# --- Extract Features ---
def extract_features(song_path):
    waveform, sr = torchaudio.load(song_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000)(waveform)
    feature_vector = torch.mean(mel_spec, dim=-1).squeeze().detach().numpy()
    feature_vector = feature_vector[:128] if feature_vector.shape[0] > 128 else np.pad(feature_vector, (0, 128 - feature_vector.shape[0]))
    return feature_vector

# --- Predict Section Label ---
LABEL_MAPPING = {
    0: 'Intro',
    1: 'Verse',
    2: 'Pre-Chorus',
    3: 'Chorus',
    4: 'Bridge',
    5: 'Breakdown',
    6: 'Instrumental',
    7: 'Outro'
}

def predict_section_label(section_feature):
    fake_logits = np.random.rand(8)  # Random prediction (no real model)
    prediction = np.argmax(fake_logits)
    return LABEL_MAPPING[prediction]

# --- Main Program ---
song_path = select_audio_file()
song_path = convert_to_wav(song_path)

# Load audio
waveform, sample_rate = torchaudio.load(song_path)
print(f"Waveform shape: {waveform.shape}")
print(f"Sample rate: {sample_rate}")

# Genre classification
genre_results = classify_genre(song_path)
print(f"The predicted genre is {genre_results[0]['label']} with {genre_results[0]['score']*100:.1f}% certainty.")

# --- Structure Detection ---
y, sr = librosa.load(song_path)
hop_length = 512

# Onset strength
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
novelty_smooth = scipy.signal.medfilt(onset_env, kernel_size=7)
peaks, _ = scipy.signal.find_peaks(novelty_smooth, distance=(sr//hop_length)*5)

boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
boundary_times = np.concatenate(([0], boundary_times, [librosa.get_duration(y=y, sr=sr)]))

# Merge small segments
min_duration = 15.0
final_times = [boundary_times[0]]
for t in boundary_times[1:]:
    if t - final_times[-1] >= min_duration:
        final_times.append(t)

# --- Predict section labels ---
labels_predicted = []
for start, end in zip(final_times[:-1], final_times[1:]):
    temp_audio, _ = librosa.load(song_path, sr=16000, offset=start, duration=end-start)
    if len(temp_audio) == 0:
        labels_predicted.append('Outro')
        continue
    temp_audio_path = "temp_section.wav"
    sf.write(temp_audio_path, temp_audio, 16000)  # <-- Correct way to write a temp wav file

    section_feature = extract_features(temp_audio_path)
    label = predict_section_label(section_feature)
    labels_predicted.append(label)

    os.remove(temp_audio_path)

# --- Output Results ---
print("\nDetected Sections:")
for i in range(len(final_times)-1):
    print(f"Section {i+1}: {final_times[i]:.2f} sec → {final_times[i+1]:.2f} sec")

print("\nPredicted Labels:")
for label, start, end in zip(labels_predicted, final_times[:-1], final_times[1:]):
    print(f"{label}: {start:.2f} sec → {end:.2f} sec")

# --- Plotting ---
plt.figure(figsize=(14, 8))
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_db = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')

for t in final_times:
    plt.axvline(x=t, color='white', linestyle='--', linewidth=2)

plt.title('Spectrogram with Detected Sections')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# --- Lua Script Generation ---
sequence = input("What grandMA sequence? (provide a number) ")
lua_script_lines = []
lua_script_lines.append("--********************************************************************")
lua_script_lines.append("--**           Lighting Cue Importer                                **")
lua_script_lines.append("--********************************************************************")
lua_script_lines.append("local CMD = gma.cmd;")
lua_script_lines.append("local ECHO = gma.echo;")
lua_script_lines.append("local FEEDBACK = gma.feedback;")
lua_script_lines.append("local DIALOG = gma.gui.confirm;")
lua_script_lines.append("local GET_OBJ = gma.show.getobj")
lua_script_lines.append("local GET_HANDLE = gma.show.getobj.handle;")
lua_script_lines.append("local GET_INDEX = gma.show.getobj.index;")
lua_script_lines.append("local GET_PROPERTY = gma.show.property.get;")
lua_script_lines.append("local GETVAR = gma.show.getvar;")
lua_script_lines.append("local ZZZ = gma.sleep;")
lua_script_lines.append("local LUA_NAME = 'Lighting Cues : ';\n")

lua_script_lines.append("--********************************************************************")
lua_script_lines.append("--**             Main Function                                      **")
lua_script_lines.append("--********************************************************************")
lua_script_lines.append("local function lightingcueimporter()")
lua_script_lines.append("\n-- Store Cues:")
cue_number = 1
for label, start, end in zip(labels_predicted, final_times[:-1], final_times[1:]):
    cue_name = f"{label} {start:.0f}-{end:.0f}"
    lua_script_lines.append(f'CMD(\'Store Sequence {sequence} Cue {cue_number} \"{cue_name}\"\')')
    lua_script_lines.append(f'CMD(\'Assign Sequence {sequence} Cue {cue_number} /Fade=3.0\')')
    cue_number += 1

lua_script = "\n".join(lua_script_lines)

lua_script_lines.append("CMD('SelectDrive 1');")
lua_script_lines.append("gma.sleep(0.5);")
lua_script_lines.append("end")
lua_script_lines.append("return lightingcueimporter;")

# Save LUA file
with open(f"cuepoints_generated.lua", "w") as f:
    f.write(lua_script)

print("\nLua script 'cuepoints_generated.lua' has been created!")