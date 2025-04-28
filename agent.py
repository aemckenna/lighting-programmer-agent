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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import datetime

# load wav2vec2 pretrained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# load audio
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

# convert to wav
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

# classify as a genre
def classify_genre(audio_path):
    pipe = pipeline("audio-classification", model="dima806/music_genres_classification")
    return pipe(audio_path)

# get audio metrics and features
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

# song section labeling
def extract_section_features(y, sr=16000):
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.sum(harmonic**2) / (np.sum(harmonic**2) + np.sum(percussive**2))
    mfcc_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    return {
        'rms': rms,
        'centroid': centroid,
        'bandwidth': bandwidth,
        'flatness': flatness,
        'zcr': zcr,
        'harmonic_ratio': harmonic_ratio,
        'mfcc_mean': mfcc_mean
    }

# identifying speech or lyrics
def detect_speech_density(audio_clip, sr=16000):
    import torch

    if sr != 16000:
        audio_clip = librosa.resample(audio_clip, orig_sr=sr, target_sr=16000)
    
    input_values = processor(audio_clip, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    words = transcription.split()
    return len(words)

# 
def predict_section_label(section_audio, song_global, genre="pop", sr=16000):
    features = extract_section_features(section_audio, sr)
    
    rms = features['rms']
    centroid = features['centroid']
    zcr = features['zcr']
    bandwidth = features['bandwidth']
    flatness = features['flatness']
    harmonic_ratio = features['harmonic_ratio']
    mfcc_mean = features['mfcc_mean']
    
    #print(f"RMS: {rms:.4f} | Centroid: {centroid:.1f} | ZCR: {zcr:.4f} | Bandwidth: {bandwidth:.1f} | Flatness: {flatness:.4f} | Harmonic Ratio: {harmonic_ratio:.4f} | MFCC Mean: {mfcc_mean:.2f}")
    
    speech_density = detect_speech_density(section_audio, sr)
    print(f"Speech Words Detected: {speech_density}")
    lyrics_present = speech_density >= 5

    scores = {}

    rms_rel = rms / song_global['rms']
    centroid_rel = centroid / song_global['centroid']
    
    if genre == "pop":
        if lyrics_present:
            scores['Chorus'] = (rms_rel > 1.2) * 1.5 + (centroid_rel > 1.1) * 1.0
            scores['Pre-Chorus'] = (rms_rel > 1.0) * 1.0 + (centroid_rel > 1.0) * 0.8
            scores['Verse'] = (rms_rel < 1.0) * 1.0 + (centroid_rel < 1.0) * 0.8
        else:
            scores['Intro'] = (rms < 0.03) * 1.0 + (harmonic_ratio > 0.7) * 0.5
            scores['Breakdown'] = (rms < 0.02) * 1.0 + (flatness > 0.3) * 0.5
            scores['Instrumental'] = (centroid < 1000) * 1.0 + (flatness > 0.3) * 0.5
            scores['Outro'] = (rms < 0.05) * 1.0 + (harmonic_ratio > 0.6) * 0.5

    else:
        if lyrics_present:
            scores['Chorus'] = (rms_rel > 1.2) * 1.0
            scores['Verse'] = (rms_rel <= 1.2) * 1.0
        else:
            scores['Outro'] = 1.0

    best_label = max(scores, key=scores.get)
    confidence = scores[best_label]

    #print(f"Predicted: {best_label} with confidence {confidence:.2f}")
    return best_label

# get audio and convert it
song_path = select_audio_file()
song_path = convert_to_wav(song_path)

# load audio
waveform, sample_rate = torchaudio.load(song_path)
print(f"Waveform shape: {waveform.shape}")
print(f"Sample rate: {sample_rate}")

# classify genre
genre_results = classify_genre(song_path)
print(f"The predicted genre is {genre_results[0]['label']} with {genre_results[0]['score']*100:.1f}% certainty.")

# structure detection
y, sr = librosa.load(song_path)
hop_length = 512
song_global_features = extract_section_features(y, sr=sr)

# onset strength
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
novelty_smooth = scipy.signal.medfilt(onset_env, kernel_size=7)
peaks, _ = scipy.signal.find_peaks(novelty_smooth, distance=(sr//hop_length)*5)

boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
boundary_times = np.concatenate(([0], boundary_times, [librosa.get_duration(y=y, sr=sr)]))

# merge small segments
min_duration = 15.0
final_times = [boundary_times[0]]
for t in boundary_times[1:]:
    if t - final_times[-1] >= min_duration:
        final_times.append(t)

# predict section labels
labels_predicted = []
for start, end in zip(final_times[:-1], final_times[1:]):
    temp_audio, _ = librosa.load(song_path, sr=16000, offset=start, duration=end-start)
    if len(temp_audio) == 0:
        labels_predicted.append('Outro')
        continue

    label = predict_section_label(temp_audio, song_global_features)
    labels_predicted.append(label)

# output all the results
print("\nDetected Sections:")
for i in range(len(final_times)-1):
    print(f"Section {i+1}: {final_times[i]:.2f} sec → {final_times[i+1]:.2f} sec")

print("\nPredicted Labels:")
for label, start, end in zip(labels_predicted, final_times[:-1], final_times[1:]):
    print(f"{label}: {start:.2f} sec → {end:.2f} sec")

# visual plotting
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

# lua script code
sequence = input("\nWhat grandMA sequence? (provide a number) ")
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
lua_script_lines.append("local LUA_NAME = 'LightingCues';\n")

lua_script_lines.append("--********************************************************************")
lua_script_lines.append("--**             Main Function                                      **")
lua_script_lines.append("--********************************************************************")
lua_script_lines.append("local function lightingcueimporter()")
lua_script_lines.append("\n-- Store Cues:")
cue_number = 1
for label, start, end in zip(labels_predicted, final_times[:-1], final_times[1:]):
    cue_name = f"{label} {start:.0f}-{end:.0f}"
    lua_script_lines.append(f'CMD(\'Store Sequence {sequence} Cue {cue_number} \"{cue_name}\"\');')
    lua_script_lines.append(f'CMD(\'Assign Sequence {sequence} Cue {cue_number} /Fade=3.0\');')
    cue_number += 1

lua_script_lines.append("CMD('SelectDrive 1');")
lua_script_lines.append("gma.sleep(0.5);")
lua_script_lines.append("end")
lua_script_lines.append("return lightingcueimporter;")

lua_script = "\n".join(lua_script_lines)

# save lua script
song_filename = os.path.basename(song_path)
song_name_without_ext = os.path.splitext(song_filename)[0]
lua_filename = f"{song_name_without_ext}.lua"

with open(lua_filename, "w") as f:
    f.write(lua_script)

# create timestamp
now = datetime.datetime.now()
timestamp_str = now.strftime("%Y-%m-%dT%H:%M:%S")

# create xml file
xml_filename = f"{song_name_without_ext}.xml"

xml_contents = f'''<?xml version="1.0" encoding="UTF-8"?>
<MA xmlns:xml="http://www.w3.org/XML/1998/namespace" xsi:schemaLocation="http://schemas.malighting.de/grandma2/xml/MA http://schemas.malighting.de/grandma2/xml/3.3.1/MA.xsd" major_vers="3" minor_vers="8" stream_vers="0">
  <Info xmlns="http://schemas.malighting.de/grandma2/xml/MA" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" datetime="{timestamp_str}" showfile="Lighting"/>
  <Plugin xmlns="http://schemas.malighting.de/grandma2/xml/MA" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" index="0" name="LightingCues {timestamp_str}" luafile="{song_name_without_ext}.lua"/>
</MA>
'''

with open(xml_filename, "w") as xml_file:
    xml_file.write(xml_contents)

print(f"\nXML file '{xml_filename}' has been created!")

print(f"\nLua script '{song_name_without_ext}.lua' has been created!")