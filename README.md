# Song Section Labeler + GrandMA2 Cue Generator

This project is designed to analyze a song, detect musical sections (such as Intro, Verse, Chorus, etc.), and generate:
- A `.lua` script for GrandMA2 to create cues for lighting sequences
- A `.xml` plugin file for importing the script into GrandMA2
- A visual plot showing the detected section boundaries

It uses audio analysis with Librosa, Torchaudio, Wav2Vec2 (speech detection), and feature analysis to make predictions about song structure, energy, and lyrics.

---

# How It Works

1. Select a song (`.wav`, `.mp3`, etc.) using a file dialog.
2. Audio is converted to `.wav`.
3. Genre classification is performed (Pop, Rock, etc.).
4. Global features of the full song are extracted.
5. Sections (Intro, Verse, Chorus, Bridge...) are predicted dynamically based on:
   - Energy (RMS)
   - Spectral Centroid (high centroid is bright and low centroid is muffled) and Bandwidth
   - Harmonic vs Percussive balance
   - Speech density (lyrics are detected using Wav2Vec2)
6. Cues are automatically named and stored based on section types.
7. A Lua script is generated for programming cues into GrandMA2.
8. An XML plugin file is generated to load the lua script into GrandMA2.

---

# Features

- Speech Detection with HuggingFace's `facebook/wav2vec2-large-960h` model
- Audio Analysis (Librosa + Torchaudio)
- Automatic Cue Naming based on song sections
- Lua + XML Generation to import placeholder cues into GrandMA2
- Takes 20-30 seconds per song.

---

# Requirements

- Python 3.8+
- `librosa`
- `torch`
- `torchaudio`
- `transformers`
- `soundfile`
- `tkinter`
- `scipy`
- `matplotlib`

Run this command to install everything:
```bash
pip install librosa torchaudio transformers soundfile scipy matplotlib
```
---

# Presentation and Example Run
[Check out the presentation/demo here!](https://www.loom.com/share/d8381b1ef71f41189c0ee36e619622a5?sid=081d6361-5ff1-4517-8862-1fe4dad6f913)

Screenshot of the program after running:
<img width="823" alt="Screenshot 2025-04-27 at 9 52 03 PM" src="https://github.com/user-attachments/assets/2a17ff31-56a4-4a39-8c7c-03cd5540c7c6" />

Outputs this graph to visually see where it thinks song section changes happen:
![Uploading Screensho![Uploading Screenshot 2025-04-27 at 2.31.24 PM.png…]()
t 2025-04-27 at 9.54.09 PM.png…]()

LUA and XML script output to a sequence pool:
<img width="1094" alt="Screenshot 2025-04-27 at 9 53 03 PM" src="https://github.com/user-attachments/assets/93acd552-d5f4-4cf6-9c49-135af9219f6e" />

Open the sequence pool to see the cues that have been created:
<img width="660" alt="Screenshot 2025-04-27 at 9 52 54 PM" src="https://github.com/user-attachments/assets/04d761b6-c097-45fe-849d-6528641ac9c6" />

---

# Future Plans

Honestly this project took a bit more thinking through than I thought it would. I feel like it is close to guessing when there are section changes in a song but sometimes it is off by a tiny bit and favors changing vocally rather than musically. Since it has a time minimum to each section it can also tend to favor 15 or 30 second chunks over the actual sections. In general though it works fairly well with pop songs and songs that have musical and energy variety.

Ways to improve this would be by nailing down a better way to predict changes such as finding a data set or training one and use that paired with the techniques that I am using now. There are some such as oh Horus that are decent at detecting choruses by looking for patterns but not great at the verse, intro, bridge, etc.. To take this project a step further I would work on implementing that more.

I would also integrate a way to create a timecode pool object and a timecode xml file so that the lighting cues created would be linked via SMPTE timecode to simplify the programming process even more. 

And lastly, if I were to take this one step further I would integrate this with some sort of agent that would use the energy and features from each section and come up with potential color schemes and lighting effects that would go along with the energy and vibe of the song.


