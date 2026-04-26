import os
import random
import pickle
import numpy as np
from flask import Flask, render_template, jsonify, send_file, request

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Speed optimization
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from music21 import note, chord, stream, instrument

app = Flask(__name__)

MODEL       = None
NOTES       = None
NOTE_TO_INT = None
INT_TO_NOTE = None
N_VOCAB     = None
SEQ_LEN     = None
PREDICT_FN  = None  # Compiled prediction function


def build_model(n_vocab, sequence_length):
    model = Sequential([
        LSTM(512, input_shape=(sequence_length, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(512, return_sequences=True),
        Dropout(0.3),
        LSTM(256, return_sequences=False),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(n_vocab, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def load_everything():
    global MODEL, NOTES, NOTE_TO_INT, INT_TO_NOTE, N_VOCAB, SEQ_LEN, PREDICT_FN
    print("=" * 45)
    print("  🎵 CodeAlpha AI Music Generator")
    print("=" * 45)
    required = ["model_weights.weights.h5", "vocab.pkl", "notes.pkl"]
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing: {missing}")
        return False

    with open("notes.pkl", "rb") as f:
        NOTES = pickle.load(f)
    print(f"✅ Notes: {len(NOTES)}")

    with open("vocab.pkl", "rb") as f:
        pitchnames, NOTE_TO_INT, INT_TO_NOTE, N_VOCAB, SEQ_LEN = pickle.load(f)
    print(f"✅ Vocab: {N_VOCAB} | SeqLen: {SEQ_LEN}")

    MODEL = build_model(N_VOCAB, SEQ_LEN)
    MODEL.load_weights("model_weights.weights.h5")

    # Warm up model — first prediction slow hoti hai
    print("⚡ Warming up model...")
    dummy = np.zeros((1, SEQ_LEN, 1))
    MODEL.predict(dummy, verbose=0)
    print("✅ Model warmed up!")

    # TF Function compile karo — fast prediction
    @tf.function(reduce_retracing=True)
    def fast_predict(x):
        return MODEL(x, training=False)

    PREDICT_FN = fast_predict
    print("✅ Fast prediction ready!")
    print("=" * 45)
    print("  🌐 Open: http://127.0.0.1:5000")
    print("=" * 45)
    return True


def generate_notes(temperature=0.8, num_notes=50):
    start   = random.randint(0, len(NOTES) - SEQ_LEN - 1)
    pattern = [NOTE_TO_INT[n] for n in NOTES[start:start + SEQ_LEN]]
    generated = []

    print(f"🎵 Generating {num_notes} notes (fast mode)...")

    for i in range(num_notes):
        # Fast prediction
        x          = np.reshape(pattern, (1, SEQ_LEN, 1)).astype(np.float32)
        x          = x / float(N_VOCAB)
        prediction = PREDICT_FN(x).numpy()[0]

        # Temperature sampling
        prediction = np.log(prediction + 1e-8) / temperature
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        idx        = np.random.choice(len(prediction), p=prediction)

        generated.append(INT_TO_NOTE[idx])
        pattern.append(idx)
        pattern = pattern[1:]

    print(f"✅ {num_notes} notes generated!")
    return generated


def notes_to_midi(prediction_output, output_file):
    offset, output_notes = 0.0, []
    for pattern in prediction_output:
        try:
            if "." in pattern:
                chord_notes = []
                for cn in pattern.split("."):
                    try:
                        n = note.Note(int(cn))
                        n.storedInstrument = instrument.Piano()
                        chord_notes.append(n)
                    except:
                        pass
                if chord_notes:
                    c        = chord.Chord(chord_notes)
                    c.offset = offset
                    output_notes.append(c)
                    offset += 0.5
            else:
                try:
                    n = note.Note(int(pattern))
                except ValueError:
                    n = note.Note(pattern)
                n.storedInstrument = instrument.Piano()
                n.offset = offset
                output_notes.append(n)
                offset += 0.5
        except:
            pass
    stream.Stream(output_notes).write("midi", fp=output_file)
    print("✅ MIDI saved!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data        = request.get_json() or {}
        temperature = float(data.get("temperature", 0.8))
        temperature = max(0.3, min(1.2, temperature))

        generated = generate_notes(temperature=temperature, num_notes=50)
        os.makedirs("static/generated", exist_ok=True)

        midi_path = "static/generated/output.mid"
        wav_path  = "static/generated/output.wav"

        notes_to_midi(generated, midi_path)

        # WAV seedha serve karo
        if os.path.exists(wav_path):
            audio_url  = "/static/generated/output.wav"
            audio_type = "wav"
            print("✅ WAV serving to browser!")
        else:
            audio_url  = "/static/generated/output.mid"
            audio_type = "midi"
            print("⚠️ WAV not found!")

        return jsonify({
            "success"   : True,
            "audio_url" : audio_url,
            "audio_type": audio_type
        })

    except Exception as e:
        print(f"❌ {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/download")
def download():
    for path, name in [
        ("static/generated/output.wav", "ai_music.wav"),
        ("static/generated/output.mid", "ai_music.mid")
    ]:
        if os.path.exists(path):
            return send_file(path, as_attachment=True, download_name=name)
    return jsonify({"error": "Generate music first!"})


if __name__ == "__main__":
    if load_everything():
        app.run(debug=False, host="0.0.0.0", port=5000)
    else:
        print("❌ Fix missing files!")