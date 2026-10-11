# Praktikum Bab 10 — Membangun Sistem RAG untuk Tanya-Jawab Dokumen Berbahasa Indonesia

Bagian dari buku *Pemrosesan Bahasa Alami: Teori dan Praktikum*, Bab 10 **Information Retrieval, Question Answering, dan RAG**.

Praktikum ini membangun satu sistem RAG utuh: memecah dokumen, membangun empat konfigurasi retriever beserta pengukurannya, menyusun generator yang memberi sitasi, lalu mengevaluasi sisi retriever dan sisi generator secara terpisah. Sistem yang dibangun di sini **dipakai lagi di praktikum Bab 12 (Agentic AI)**.

## Isi

| Berkas | Keterangan |
|---|---|
| `template_rag.ipynb` | Kerangka kode. Bagian bertanda `TODO` dilengkapi praktikan. |
| `solusi_rag.ipynb` | Satu contoh pengerjaan lengkap, sebagai pembanding. |

Kedua notebook **berdiri sendiri**. Korpus contoh, dataset uji, dan fungsi pemanggil LLM sudah ada di dalam notebook, jadi tidak perlu mengunduh berkas lain. Notebook solusi bukan satu-satunya jawaban benar: praktikum boleh dikerjakan dengan model dan strategi lain, asalkan pengukurannya memakai dataset uji yang sama.

## Langkah praktikum

| Langkah | Isi | Bagian `TODO` di template |
|---|---|---|
| 1 | Persiapan lingkungan dan verifikasi (pustaka, model embedding, LLM) | – |
| 2 | Korpus dan dataset uji | – |
| 3 | Chunking (fixed-size, recursive), embedding E5, ChromaDB | `chunk_fixed`, `chunk_recursive` |
| 4 | Retrieval: dense, BM25 (dengan dan tanpa Sastrawi), hybrid RRF, hybrid + rerank; Recall@5, Recall@10, MRR | `dense`, `bm25`, `hybrid`, `hybrid_rerank` |
| 5 | Generator RAG dengan sitasi, penolakan untuk pertanyaan tak terjawab, eksperimen *Lost in the Middle* | aturan di `PROMPT_RAG` |
| 6 | Evaluasi generator: faithfulness, answer relevance, uji keandalan juri LLM | `pecah_klaim`, `cek_klaim`, `relevansi` |
| 7 | Versi framework dengan LangChain (environment terpisah) | – |

## Persiapan

### 1. Python dan virtual environment

Gunakan **Python 3.10 atau 3.11**. Versi yang lebih baru sering bermasalah dengan pustaka machine learning.

```bash
python3.11 -m venv nlp-rag-env
source nlp-rag-env/bin/activate        # Linux/Mac
nlp-rag-env\Scripts\activate           # Windows
```

### 2. Pustaka

```bash
pip install --upgrade pip
pip install sentence-transformers==3.0.1 chromadb==0.5.3 rank-bm25==0.2.2 numpy==1.26.4 pandas==2.2.2 \
            scikit-learn==1.5.1 jupyter==1.0.0 python-dotenv==1.0.1 ollama==0.3.3 "google-genai>=1.0" \
            Sastrawi==1.0.1 "posthog<6"
```

- Nama pustaka Gemini adalah **`google-genai`**, bukan `google-generativeai` (SDK lama yang sudah tidak dikembangkan).
- `posthog<6` mencegah pesan galat telemetri ChromaDB 0.5.3 (*"Failed to send telemetry event"*).
- Dengan GPU NVIDIA, pasang PyTorch versi CUDA lebih dulu agar embedding dan reranking lebih cepat (lihat https://pytorch.org/get-started/locally/).

### 3. LLM

Pilih salah satu jalur, atau siapkan keduanya sebagai cadangan.

**Jalur A — Ollama (lokal, tanpa biaya).** Pasang dari https://ollama.com/download, lalu:

```bash
ollama pull qwen2.5:7b-instruct      # butuh RAM sekitar 8 GB; alternatif: qwen2.5:3b-instruct
```

**Jalur B — Gemini API (kuota gratis).** Buat API key di https://aistudio.google.com (menu *Get API key*). Daftar model yang tersedia berubah cukup sering, jadi periksa https://ai.google.dev/gemini-api/docs/models.

### 4. Berkas `.env`

Buat berkas `.env` di folder yang sama dengan notebook:

```
LLM_BACKEND=ollama               # atau: gemini
LLM_MODEL=qwen2.5:7b-instruct    # atau: gemini-2.5-flash
GEMINI_API_KEY=                  # hanya untuk jalur Gemini
RPM_MAKS=10                      # batas permintaan per menit untuk API berkuota

# Opsional
EMBED_MODEL=intfloat/multilingual-e5-small
RERANK_MODEL=BAAI/bge-reranker-v2-m3
# RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1   # lebih ringan untuk CPU
```

**Jangan pernah** menulis API key di dalam sel notebook. Pastikan `.env` tercantum di `.gitignore`. Key yang pernah ter-commit ke Git harus dianggap bocor dan segera dicabut.

## Menjalankan

```bash
jupyter notebook template_rag.ipynb
```

Jalankan sel dari atas ke bawah. Sel verifikasi di Langkah 1 memeriksa pustaka, model embedding, dan LLM; jangan lanjut sebelum ketiganya lolos.

Saat eksperimen, isi `SUBSET = 8` di sel konfigurasi agar hemat kuota. Kembalikan ke `None` untuk pengukuran akhir.

### Struktur folder setelah dijalankan

```
.
├── template_rag.ipynb
├── solusi_rag.ipynb
├── .env
├── data/
│   ├── korpus/                 # 6 dokumen peraturan akademik fiktif (dibuat otomatis)
│   └── dataset_uji.json        # 32 pertanyaan uji (dibuat otomatis)
├── indeks/                     # ChromaDB: korpus_fixed, korpus_recursive, korpus_recursive_tanpa_awalan
└── hasil/
    ├── retrieval.json          # Langkah 4
    ├── jawaban_rag.json        # Langkah 5
    ├── evaluasi_generator.json # Langkah 6
    └── penilaian_manual.csv    # Langkah 6.3, diisi manual
```

Simpan folder `data/`, `indeks/`, dan `hasil/`; semuanya dipakai lagi di praktikum Bab 12.

## Data

**Korpus contoh**: enam dokumen peraturan akademik dari kampus **fiktif** "Institut Teknologi Sagara" (peraturan akademik umum, cuti, tugas akhir, kelulusan, penilaian, dan FAQ layanan). Untuk memakai korpus asli, letakkan berkas `.txt` (UTF-8) di `data/korpus/`. Notebook tidak menimpa folder yang sudah berisi.

**Dataset uji** (`data/dataset_uji.json`): 32 pertanyaan dalam empat kategori.

| Tipe | Jumlah | Yang diuji |
|---|---|---|
| `istilah_spesifik` | 10 | Kekuatan BM25 (kode, angka, nama yang persis ada di dokumen) |
| `parafrase` | 8 | Kekuatan dense retrieval (kata berbeda dari dokumen) |
| `multi_hop` | 8 | Jawaban dirakit dari dua dokumen; kelak kemampuan agent |
| `tidak_terjawab` | 6 | Keberanian sistem menolak menjawab |

Format satu entri:

```json
{
  "id": "Q01",
  "pertanyaan": "Berapa SKS minimum untuk mengambil tugas akhir?",
  "jawaban_acuan": "110 SKS (syarat mengambil IF4091 Tugas Akhir I).",
  "dokumen_kunci": ["peraturan_tugas_akhir.txt"],
  "tipe": "istilah_spesifik"
}
```

Dataset uji sebaiknya disiapkan dan divalidasi asisten praktikum agar angka antar-kelompok dapat dibandingkan.

## Langkah 7: LangChain

Langkah 7 memakai versi chromadb dan numpy yang berbeda dari Langkah 1–6, jadi pasang di **environment terpisah**:

```bash
python3.11 -m venv nlp-langchain-env
source nlp-langchain-env/bin/activate
pip install langchain-core==1.6.9 langchain-community==0.4.2 langchain-text-splitters==1.1.3 \
            langchain-huggingface==1.2.3 langchain-chroma==1.1.0 langchain-ollama==1.1.0 \
            langchain-google-genai==4.4.1 sentence-transformers==6.1.0 python-dotenv jupyter
```

Buka notebook dengan kernel environment ini, lalu jalankan **hanya** sel-sel di bagian Langkah 7. Sel tersebut membaca `data/` dan `hasil/retrieval.json` dari Langkah 1–6 untuk membandingkan Recall@5.

## Mengelola kuota API

Kuota gratis Gemini dibatasi per menit (RPM, TPM) dan per hari (RPD). Praktikum ini bisa memakai beberapa ratus panggilan LLM.

1. Kembangkan dengan Ollama, lalu jalankan pengukuran akhir dengan Gemini.
2. Kecilkan `SUBSET` saat eksperimen.
3. Sesuaikan `RPM_MAKS` dengan batas di dashboard AI Studio. Fungsi `chat()` sudah menjeda antar-panggilan dan mencoba ulang otomatis saat kena galat 429.
4. Hasil setiap langkah tersimpan di `hasil/`, jadi langkah yang sudah selesai tidak perlu diulang.

Kuota harian disetel ulang tengah malam waktu Pasifik, sekitar pukul 14.00–15.00 WIB.

## Pengumpulan

1. Notebook yang sudah dilengkapi dan bisa dijalankan ulang dari atas ke bawah tanpa galat.
2. Folder `hasil/`.
3. Laporan berisi jawaban Tugas Langkah 3 sampai 7.

**Syarat reproduktivitas:** temperature 0 untuk semua pemanggilan yang dievaluasi; seed dicatat; versi model dan pustaka dicantumkan; `.env` tidak ikut dikumpulkan. Bila skor berubah antar-jalankan, laporkan rentangnya, bukan hanya satu angka.

### Rubrik penilaian

| Komponen | Bobot | Kriteria |
|---|---|---|
| Sistem berjalan | 20% | Notebook berjalan dari atas ke bawah, keluaran tersimpan |
| Kelengkapan implementasi | 25% | Dua metode chunking, empat konfigurasi retriever, generator dengan sitasi, faithfulness, versi LangChain |
| Kebenaran pengukuran | 15% | Metrik dihitung benar, perbandingan memakai dataset uji yang sama |
| **Analisis kegagalan** | **30%** | Setiap tugas disertai contoh kasus gagal dan penjelasan penyebabnya |
| Laporan | 10% | Tabel lengkap, tulisan jelas, kesimpulan didukung angka |

## Troubleshooting

| Gejala | Kemungkinan penyebab | Penanganan |
|---|---|---|
| `ollama: connection refused` | Layanan Ollama belum berjalan | Jalankan `ollama serve` di terminal terpisah |
| `429 RESOURCE_EXHAUSTED` | Batas kuota Gemini tercapai | Turunkan `RPM_MAKS`, kecilkan `SUBSET`, pakai varian Flash-Lite, atau tunggu kuota pulih |
| `API key not valid` | Key salah salin atau sudah dicabut | Buat key baru; pastikan `.env` ada di folder notebook |
| `404 model not found` | Nama model sudah tidak tersedia | Periksa daftar model terkini, ganti `LLM_MODEL` |
| `ModuleNotFoundError: google.generativeai` | Mengikuti tutorial SDK lama | Praktikum ini memakai `google-genai` |
| `Failed to send telemetry event` | posthog versi baru dengan chromadb 0.5.3 | `pip install "posthog<6"` |
| Unduhan model embedding gagal | Jaringan memblokir Hugging Face | Unduh di jaringan lain, simpan lokal, isi `EMBED_MODEL` dengan path folder |
| ChromaDB error saat `add` | ID chunk tidak unik | ID dibentuk dari nama dokumen dan nomor urut |
| Hasil dense kacau | Awalan `query:`/`passage:` tidak dipakai | Model E5 wajib memakai awalan |
| Reranking sangat lambat | Reranker besar berjalan di CPU | Pakai `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` atau kurangi kandidat |
| Model menjawab tanpa sitasi | Model kecil kurang patuh instruksi | Tambahkan contoh jawaban bersitasi di prompt, atau naikkan ukuran model |
| `ImportError` di Langkah 7 | Versi LangChain bentrok | Pakai environment terpisah sesuai petunjuk di atas |

## Lanjut ke Bab 12

Di praktikum Bab 12, retriever terbaik dari Langkah 4 dibungkus menjadi tool `cari_dokumen(query)` yang dipanggil oleh agent. Letakkan `template_agentic.ipynb` di folder yang sama agar indeks dan dataset uji dari praktikum ini langsung dipakai ulang.
