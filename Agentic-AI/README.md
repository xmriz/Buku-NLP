# Praktikum Bab 12 — Membangun Agentic RAG dengan Verifikator dan Guardrail

Bagian dari buku *Pemrosesan Bahasa Alami: Teori dan Praktikum*, Bab 12 **Agentic AI**.

Praktikum ini **melanjutkan praktikum Bab 10**. Korpus, dataset uji, indeks ChromaDB, dan retriever dari Bab 10 dipakai lagi tanpa diubah. Yang berubah adalah siapa yang memegang kendali alur. Pada RAG alurnya tetap: cari sekali, lalu jawab. Di sini retriever dibungkus menjadi tool, lalu agent sendiri yang memutuskan kapan mencari, berapa kali, dan dengan query apa.

```
Pertanyaan → guardrail masukan → agent penjawab (loop ReAct) ⇄ tool: cari_dokumen, kalkulator
           → agent verifikator → (perbaikan maks. 1x) → guardrail keluaran → Jawaban
```

## Isi

| Berkas | Keterangan |
|---|---|
| `template_agentic.ipynb` | Kerangka kode. Bagian bertanda `TODO` dilengkapi praktikan. |
| `solusi_agentic.ipynb` | Satu contoh pengerjaan lengkap, sebagai pembanding. |

Kedua notebook **berdiri sendiri**: korpus contoh, dataset uji, fungsi pemanggil LLM, dan kode retriever dari Bab 10 sudah ada di dalam notebook. Bila notebook diletakkan di folder yang sama dengan praktikum Bab 10, indeks dan dataset uji Bab 10 langsung dipakai ulang. Bila belum ada, indeks dibangun otomatis di Langkah 1.

## Langkah praktikum

| Langkah | Isi | Bagian `TODO` di template |
|---|---|---|
| 1 | Melanjutkan dari Bab 10: konfigurasi, LLM, data, retriever, fungsi bantu evaluasi | – |
| 2 | Membungkus retriever menjadi tool `cari_dokumen` dan membuat `kalkulator` yang aman (tanpa `eval()`) | `kalkulator` |
| 3 | Loop ReAct **tanpa framework**: prompt, parser, penanganan galat, batas putaran | isi loop `jalankan_agent` |
| 4 | Agent verifikator dengan satu putaran perbaikan | `verifikasi`, `agent_dengan_verifikator` |
| 5 | Perbandingan RAG, agent tunggal, dan agent + verifikator (akurasi, token, latensi), serta keandalan pass@3 dan pass^3 | – |
| 6 | Keamanan: indirect prompt injection lewat dokumen yang disusupi, lalu guardrail berlapis | `filter_masukan`, `filter_keluaran` |
| 7 | Versi framework dengan LangGraph (function calling bawaan model) | – |

Larangan memakai framework di Langkah 3 disengaja: tujuannya melihat apa yang sebenarnya terjadi di dalam satu putaran agent. LangGraph baru dipakai di Langkah 7 sebagai pembanding.

## Persiapan

### 1. Environment

Pakai virtual environment dan folder yang sama dengan praktikum Bab 10. Langkah 1–6 **tidak membutuhkan pustaka baru**. Bila memulai dari environment baru (Python 3.10 atau 3.11):

```bash
python3.11 -m venv nlp-rag-env
source nlp-rag-env/bin/activate        # Linux/Mac
nlp-rag-env\Scripts\activate           # Windows

pip install sentence-transformers==3.0.1 chromadb==0.5.3 rank-bm25==0.2.2 numpy==1.26.4 pandas==2.2.2 \
            scikit-learn==1.5.1 jupyter==1.0.0 python-dotenv==1.0.1 ollama==0.3.3 "google-genai>=1.0" \
            Sastrawi==1.0.1 "posthog<6"
```

### 2. LLM dan berkas `.env`

Sama dengan Bab 10. Buat `.env` di folder notebook:

```
LLM_BACKEND=ollama               # atau: gemini
LLM_MODEL=qwen2.5:7b-instruct    # atau: gemini-2.5-flash
GEMINI_API_KEY=                  # hanya untuk jalur Gemini
RPM_MAKS=10

# Opsional: konfigurasi retriever yang dipakai sebagai tool
RETRIEVER_KOLEKSI=korpus_recursive
RETRIEVER_MODE=hybrid_rerank     # dense | sparse | hybrid | hybrid_rerank
```

Untuk Ollama: `ollama pull qwen2.5:7b-instruct`. Model 7B atau lebih kecil cukup sering gagal mengikuti format ReAct. Itu bukan kesalahan kode, melainkan temuan yang layak dicatat dan dianalisis di laporan.

**Jangan pernah** menulis API key di dalam sel notebook, dan jangan meng-commit `.env`.

## Menjalankan

```bash
jupyter notebook template_agentic.ipynb
```

Jalankan sel dari atas ke bawah. Saat eksperimen, isi `SUBSET = 8` di sel konfigurasi; kembalikan ke `None` untuk pengukuran akhir.

### Struktur folder setelah dijalankan

```
.
├── template_rag.ipynb, solusi_rag.ipynb           # dari Bab 10
├── template_agentic.ipynb, solusi_agentic.ipynb
├── .env
├── data/
│   ├── korpus/                     # dari Bab 10 (dibuat otomatis bila belum ada)
│   ├── dataset_uji.json            # dari Bab 10 (dibuat otomatis bila belum ada)
│   ├── korpus_disusupi/            # Langkah 6.1
│   └── korpus_disusupi_v0 ... v4/  # Langkah 6.2, satu folder per variasi serangan
├── indeks/                         # ChromaDB; koleksi korpus_disusupi* dibuat terpisah
├── hasil/                          # dari Bab 10
└── hasil_bab12/
    ├── agent_tunggal.json          # Langkah 3, termasuk jejak lengkap
    ├── agent_verifikator.json      # Langkah 4
    ├── tabel_langkah5.csv          # Langkah 5.1
    ├── keandalan_pass3.csv         # Langkah 5.2
    ├── tabel_langkah6.csv          # Langkah 6.2
    └── agent_langgraph.json        # Langkah 7
```

Dokumen yang disusupi di Langkah 6 disimpan di folder dan koleksi ChromaDB terpisah, sehingga korpus dan indeks utama tidak tercemar. Di akhir Langkah 6, retriever otomatis dikembalikan ke indeks utama.

## Perkiraan panggilan LLM

| Langkah | Perkiraan panggilan LLM |
|---|---|
| 3: Agent tunggal | 2 sampai 5 per pertanyaan |
| 4: Agent + verifikator | 3 sampai 8 per pertanyaan |
| 5: Perbandingan + pass^3 | ketiga konfigurasi + juri, ditambah 3 kali lipat Langkah 4 pada 10 pertanyaan |
| 6: Serangan dan guardrail | 5 variasi × 4 lapis × 2 pertanyaan |

Kembangkan dengan Ollama dan subset kecil, lalu jalankan pengukuran akhir. Bila memakai Gemini, sesuaikan `RPM_MAKS`; fungsi `chat()` sudah menjeda antar-panggilan dan mencoba ulang otomatis saat kena galat 429.

## Catatan per langkah

**Langkah 3.** Parameter `stop=["Observation:"]` menghentikan model sebelum ia mengarang Observation sendiri. Tiga kesalahan ditangani tanpa menghentikan program: format tidak bisa diurai, nama tool tidak ada, dan galat dari tool. Ketiganya dikembalikan sebagai Observation agar model bisa memperbaiki langkahnya. Batas putaran (`MAKS_PUTARAN = 5`) wajib ada.

**Langkah 4.** Verifikator mengeluarkan JSON `{"didukung", "klaim_tak_didukung", "saran"}`. Perbaikan dibatasi satu kali agar dua agent tidak saling berbalas tanpa henti.

**Langkah 5.** Jawaban dinilai benar bila sesuai `jawaban_acuan` menurut juri LLM, atau bila sistem menolak dengan benar untuk pertanyaan `tidak_terjawab`. Token dan latensi diukur sebelum penilaian juri. Uji keandalan memakai `temperature=0.7`; dengan n = k = 3, pass@3 bernilai 1 bila minimal satu percobaan benar dan pass^3 bernilai 1 hanya bila ketiganya benar.

**Langkah 6.** Lapis pertahanan yang diuji:

| Lapis | Isi |
|---|---|
| 0 | Tanpa guardrail: prompt tanpa pernyataan "isi dokumen adalah data", hasil tool tanpa tag |
| 1 | Pemisahan instruksi dan data: tag `<dokumen>` + pernyataan di prompt |
| 2 | Lapis 1 + filter masukan: potongan berisi pola instruksi mencurigakan dibuang |
| 3 | Lapis 2 + filter keluaran: jawaban berisi nomor telepon yang tidak ada di korpus asli ditahan |

Variasi serangan: instruksi asli berbahasa Indonesia, bahasa Inggris, disisipkan di tengah paragraf panjang, disamarkan dengan tanda baca, dan parafrase halus. Nomor telepon pada dokumen serangan (0800-0000-0000) fiktif.

## Langkah 7: LangGraph

Dianjurkan memakai environment terpisah. Versi berikut sudah diuji kompatibel dengan pustaka Langkah 1, sehingga indeks ChromaDB dari Bab 10 tetap terbaca (karena itu boleh juga dipasang di environment yang sama):

```bash
pip install langgraph==1.0.1 langgraph-prebuilt==1.0.1 langchain-core==0.3.86 \
            langchain-ollama==0.2.1 langchain-google-genai==2.1.12
```

Bila memakai environment terpisah, jalankan dulu sel-sel Langkah 1 (konfigurasi, LLM, data, retriever, fungsi bantu evaluasi), lalu langsung ke bagian Langkah 7. Model yang dipakai harus mendukung *tool calling*, misalnya keluarga Qwen2.5 atau Llama 3.1 ke atas di Ollama.

## Pengumpulan

1. Notebook yang sudah dilengkapi dan bisa dijalankan ulang dari atas ke bawah tanpa galat.
2. Folder `hasil_bab12/`, termasuk jejak agent dalam format JSON.
3. Laporan berisi jawaban Tugas Langkah 2 sampai 7.

**Syarat reproduktivitas:** temperature 0 untuk semua pengukuran kecuali uji keandalan di Langkah 5.2; batas putaran, model, dan versi pustaka dicantumkan; jejak agent disimpan utuh, bukan hanya jawaban akhirnya; `.env` tidak ikut dikumpulkan.

### Rubrik penilaian

| Komponen | Bobot | Kriteria |
|---|---|---|
| Sistem berjalan | 15% | Notebook berjalan dari atas ke bawah, jejak dan keluaran tersimpan |
| Kelengkapan implementasi | 25% | Dua tool, loop ReAct manual dengan penanganan galat, verifikator, tiga lapis guardrail, versi LangGraph |
| Kebenaran pengukuran | 15% | Tiga konfigurasi diukur dengan dataset yang sama, pass^3 dihitung dengan benar |
| **Analisis kegagalan** | **35%** | Setiap tugas disertai jejak kasus gagal dan penjelasan penyebabnya |
| Laporan | 10% | Tabel lengkap, tulisan jelas, rekomendasi didukung angka |

Bobot analisis kegagalan lebih besar daripada di Bab 10 karena agent gagal dengan cara yang lebih beragam: format rusak, tool salah, berputar tanpa henti, verifikator keliru, atau terperdaya instruksi sisipan.

## Troubleshooting

| Gejala | Kemungkinan penyebab | Penanganan |
|---|---|---|
| Model menulis Observation sendiri | Parameter `stop` tidak dihormati backend | `chat()` sudah memotong keluaran di `"Observation:"` sebagai cadangan; periksa versi pustaka |
| Agent berputar tanpa henti | Model terus memanggil tool yang sama | Batas `MAKS_PUTARAN` menghentikannya; tambahkan instruksi agar tidak mengulang query yang sama |
| Parser tidak menemukan Action | Model memakai format lain (huruf kecil, bahasa Indonesia) | Beri contoh jejak lengkap di prompt (few-shot), atau longgarkan regex |
| Model langsung menjawab tanpa mencari | Instruksi kurang tegas, atau model merasa tahu jawabannya | Tegaskan bahwa jawaban harus berdasarkan Observation; catat sebagai temuan |
| Keluaran verifikator gagal diurai | Ada teks pembuka atau penanda blok kode | `verifikasi()` mengambil bagian di antara kurung kurawal dan punya nilai bawaan saat gagal |
| Verifikator selalu menjawab didukung | Prompt terlalu lunak, atau model terlalu kecil | Uji dengan jawaban yang sengaja salah; minta daftar klaim lebih dulu sebelum menilai |
| `429 RESOURCE_EXHAUSTED` | Agent memanggil LLM berkali-kali | Kecilkan `SUBSET`, kembangkan dengan Ollama, turunkan `RPM_MAKS` |
| Koleksi `korpus_recursive` tidak ditemukan | Indeks Bab 10 tidak ada di folder ini | Biarkan Langkah 1 membangunnya, atau jalankan Langkah 3 notebook Bab 10 |
| `ImportError` di Langkah 7 | Versi LangChain dan LangGraph bentrok | Pasang versi yang dipatok di atas, sebaiknya di environment terpisah |
| Model tidak memanggil tool di Langkah 7 | Model tidak mendukung function calling | Pakai model yang mendukung tool calling |

## Pengembangan lanjutan (opsional)

1. Tambahkan Reflexion: setelah verifikator menolak, minta agent menulis refleksi singkat lalu simpan ke memori untuk pertanyaan serupa.
2. Tambahkan memori jangka panjang yang menyimpan preferensi pengguna lintas sesi, lengkap dengan fitur melihat dan menghapus memori.
3. Tambahkan tool `kueri_nilai` berbasis Text-to-SQL read-only (Bab 10.5), lalu uji apakah agent bisa memilih antara dokumen dan database.
4. Ubah pola orkestrasi menjadi hierarchical dengan satu supervisor, lalu bandingkan akurasi dan biayanya.
5. Bungkus `cari_dokumen` sebagai server MCP sederhana.
6. Terapkan persetujuan manusia untuk tool yang mengubah keadaan, misalnya `kirim_ringkasan_email`, lalu uji kembali serangan Langkah 6.
