happy = {
    "Bagus": 1,
    "bermanfaat": 1,
    "berkualitas": 1,
    "memenuhi ekspektasi": 1,
    "membantu": 1,
    "mudah": 1,
    "praktis": 1,
    "terpercaya": 1,
    "aman": 1,
    "lancar": 1,
    "suka": 1,
    "kece": 1,
    "mantap": 1,
    "senang": 1,
    "terjamin": 1,
    "good": 1,
    "cepat": 1,
    "tidak ribet": 1,
    "luar biasa": 1,
    "gampang": 1,
    "top": 1,
    "oke": 1,
    "berguna": 1,
    "nyaman": 1,
    "keren": 1,
    "baik": 1,
    "puas": 1,
    "berhasil": 1,
    "untung": 1,
    "nice": 1,
    "excellent": 1,
    "terbaik": 1,
    "bravo": 1,
    "best": 1,
    "ok": 1,
    "alhamdulillah": 1,
}
sad = {
    "gagal": -1,
    "susah": -1,
    "gak bisa": -1,
    "jaringan": -1,
    "tidak bisa masuk": -1,
    "gagal koneksi": -1,
    "sedih": -1,
    "tidak bisa": -1,
    "gangguan": -1,
    "shock": -1,
    "hang": -1,
    "layar kosong": -1,
    "mengecewakan": -1,
    "ga efektif": -1,
    "ga ngerti": -1,
    "lelet": -1,
}
angry = {
    "banyak trouble": -1,
    "masalah": -1,
    "stress": -1,
    "sulit": -1,
    "susahnya minta ampun": -1,
    "selalu gagal": -1,
    "uninstall": -1,
    "parah": -1,
    "menyusahkan": -1,
    "ga usah pake": -1,
    "konyol": -1,
    "komplain": -1,
    "abal-abal": -1,
    "aneh": -1,
    "buruk": -1,
    "jengkel": -1,
    "sebal": -1,
    "marah": -1,
    "emosi": -1,
    "hancur": -1,
    "payah": -1,
    "ga guna": -1,
    "ribet": -1,
    "rumit": -1,
    "lemot(lambat)": -1,
    "ga karuan": -1,
    "cacat": -1,
    "capek": -1,
    "ga becus": -1,
    "bangsat": -1,
    "sampah": -1,
    "malas": -1,
    "miris": -1,
}
fear = {
    "tidak jadi": -1,
    "bocor": -1,
    "error": -1,
    "terblokir": -1,
    "ga aman": -1,
    "tidak aman": -1,
    "reset": -1,
    "root": -1,
    "data hilang": -1,
    "takut": -1,
    "hilang": -1,
    "saldo minus": -1,
    "awas": -1,
    "hati-hati": -1,
    "penipu": -1,
    "rugi": -1,
}


def emotion_analyst(text):
    score = 0
    for word in text:
        if word in happy:
            score += happy[word]
        if word in sad:
            score += sad[word]
        if word in angry:
            score += angry[word]
        if word in fear:
            score += fear[word]

    if score >= 0:
        emotion = "happy"
    elif -2 <= score < 0:
        emotion = "sad"
    elif -5 <= score < -2:
        emotion = "angry"
    else:
        emotion = "fearful"
    return score, emotion
import pandas as pd

dataset = pd.read_csv('dataset.csv')
dataset['text_preprocessed'] = dataset['text_preprocessed'].astype(str)

results = dataset['text_preprocessed'].apply(emotion_analyst)
results = list(zip(*results))
dataset['emotion_score'] = results[0]
dataset['emotion'] = results[1]

dataset.to_csv(r'emotion.csv', index=False, header= True, index_label=None)