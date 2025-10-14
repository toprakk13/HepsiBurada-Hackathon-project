import pandas as pd
import re
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import Counter
from rapidfuzz import process

# 1️⃣ Adres normalizasyonu
def normalize_address(text):
    text = unidecode(text.lower())
    text = re.sub(r'\bmah\.?\b', 'mahallesi', text)
    text = re.sub(r'\bmh\.?\b', 'mahallesi', text)
    text = re.sub(r'\bsok\.?\b', 'sokak', text)
    text = re.sub(r'\bsk\.?\b', 'sokak', text)
    text = re.sub(r'\bno\.?\b', 'numara', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2️⃣ Verileri yükle
train = pd.read_csv("train.csv")  # id, address, label
test = pd.read_csv("test.csv")    # id, address

df = pd.read_csv("Mahalle_Listesi.csv", sep=";", header=None)

# Örnek dosyada il, ilce, mahalle bilgisi 3. ve 5. sütunlarda olabilir
# Önce gerekli sütunları alalım
df = df[[3, 5]].copy()
df.columns = ["mahalle", "raw_location"]

# raw_location sütununu '->' ile il ve ilceye ayır
df[["il", "ilce", "_"]] = df["raw_location"].str.split("->", expand=True)

# string tipine çevir, küçük harfe çevir ve Türkçe karakterleri ASCII'ye dönüştür
for col in ["il", "ilce", "mahalle"]:
    df[col] = df[col].astype(str)  # NaN veya float varsa stringe çevir
    df[col] = df[col].str.lower()  # küçük harfe çevir
    df[col] = df[col].apply(unidecode)  # Türkçe karakterleri ASCII'ye çevir

# ilce sütunundaki "-il merkezi" veya "il-merkezi" ifadelerini "merkez" olarak değiştir
df["ilce"] = df["ilce"].str.replace(r'(-il merkezi|il-merkezi)', 'merkez', regex=True)

# mahalle sütunundaki "-il merkezi" veya "il-merkezi" ifadelerini sil
df["mahalle"] = df["mahalle"].str.replace(r'(-il merkezi|il-merkezi)', '', regex=True)

# mahalle sütunundaki gereksiz noktalama ve fazla boşlukları temizle
df["mahalle"] = df["mahalle"].str.replace(r'[^\w\s]', '', regex=True)
df["mahalle"] = df["mahalle"].str.strip()
df["il"] = df["il"].str.strip()
df["ilce"] = df["ilce"].str.strip()

# sadece il, ilce, mahalle sütunlarını al
df_final = df[["il", "ilce", "mahalle"]]

# CSV olarak kaydet
df_final.to_csv("temiz_adresler.csv", index=False, encoding="utf-8")
df_clean = pd.read_csv("temiz_adresler.csv")




# 3️⃣ Clean addresses
train['clean_address'] = train['address'].apply(normalize_address)
test['clean_address'] = test['address'].apply(normalize_address)

# 4️⃣ NER setup
model_name = "dbbiyte/histurk-BERTurk-sentiment"
token="token key"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForTokenClassification.from_pretrained(model_name, use_auth_token=token)
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 5️⃣ TF-IDF + k-NN setup
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),
    max_features=25000,
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(train['clean_address'])
train_vectors = X_train_tfidf.toarray().astype('float32')
d = train_vectors.shape[1]
index = faiss.IndexFlatL2(d)
index.add(train_vectors)
labels = train['label'].values

# 6️⃣ Tahmin pipeline
predicted_labels = []

for address in test['clean_address']:
    # --- NER ile varlık çıkar ---
    entities = nlp_ner(address)
    mahalle_name = None
    for e in entities:
        if e['entity_group'] in ['LOC', 'MISC']:  # mahalle/cadde isimleri
            mahalle_name = unidecode(e['word'].lower())
            break

    # --- Canonical listeden eşle ---
    nn_label = None
    if mahalle_name:
        best_match = process.extractOne(mahalle_name, df_clean['mahalle'].tolist())
        if best_match:
            nn_label = best_match[0]

    # --- TF-IDF + k-NN ile tahmin ---
    query_vector = vectorizer.transform([address]).toarray().astype('float32')
    k = 3
    distances, indices = index.search(query_vector, k)
    top_k_labels = train['label'].iloc[indices[0]].values
    knn_label = Counter(top_k_labels).most_common(1)[0][0]

    # --- Hibrit: NER > Canonical > k-NN ---
    final_label = nn_label or knn_label
    predicted_labels.append(final_label)

# 7️⃣ Tahminleri test dataframe'e ekle
test['predicted_label'] = predicted_labels

# 8️⃣ Submission dosyası oluştur
submission = test[['id', 'predicted_label']]
submission.to_csv("submission.csv", index=False, encoding='utf-8')
print("Tahminler submission.csv dosyasına kaydedildi.")