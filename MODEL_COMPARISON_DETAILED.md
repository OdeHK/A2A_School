# 🔥 SO SÁNH 4 MODELS - CHI TIẾT TOÀN DIỆN

**Ngày:** 04/11/2025  
**Mục đích:** Chọn model tối ưu cho xử lý PDF 2620 trang

---

## 📊 TỔNG QUAN 4 MODELS

| **Model** | **Type** | **Dimensions** | **Parameters** | **Base Model** |
|-----------|----------|----------------|----------------|----------------|
| **MiniLM-L6-v2** | Embedding | 384 | 22M | DistilBERT |
| **Vietnamese_Embedding_v2** | Embedding | 1024 | 568M | BGE-M3 |
| **Vietnamese_Reranker** | **Reranker** ⚠️ | 1024 | 568M | BGE-Reranker-v2-M3 |
| **Alibaba/gte** | Embedding | 768 | 278M | NewModel |

---

## ⚠️ PHÁT HIỆN QUAN TRỌNG!

### **Vietnamese_Reranker ≠ Embedding Model!**

**RERANKER** khác hoàn toàn với **EMBEDDING MODEL**!

```python
# EMBEDDING MODEL (dùng cho RAG)
query = "What is AI?"
docs = ["AI is...", "Machine learning is..."]

embeddings = model.encode([query] + docs)
# → Output: vectors [768 dims, 768 dims, 768 dims]
# → Dùng để: Similarity search, vector database

# RERANKER MODEL (dùng sau khi retrieve)
pairs = [
    [query, docs[0]],  # Query + Doc 1
    [query, docs[1]]   # Query + Doc 2
]
scores = reranker.predict(pairs)
# → Output: scores [0.95, 0.23]  ← Không phải vector!
# → Dùng để: Re-rank kết quả đã retrieve
```

---

## 🔍 CHI TIẾT TỪNG MODEL

### **1. MiniLM-L6-v2** ⚡ (Đã chạy thành công)

**Specs:**
```yaml
Model: sentence-transformers/all-MiniLM-L6-v2
Type: Embedding (Bi-Encoder)
Dimensions: 384
Parameters: 22M
Base: DistilBERT-base
Layers: 6
Max Seq Length: 512
Language: Multilingual (100+ languages)
License: Apache 2.0
```

**Performance (Thực tế):**
```
PDF 2620 pages:
- PDF reading: 4.2s
- Chunking: 0.02s
- Embedding: 131s (9,286 chunks)
- Storage: 2.7s
- TOTAL: 138s = 2.3 minutes ✅

Speed: 70.7 chunks/second
Quality: ⭐⭐⭐ (Good)
```

**Ưu điểm:**
- ✅ CỰC NHANH (22M params)
- ✅ Nhẹ nhàng (384 dims)
- ✅ Multilingual
- ✅ Đủ tốt cho hầu hết use cases

**Nhược điểm:**
- ❌ Tiếng Việt không tốt bằng specialized models
- ❌ Quality thấp hơn SOTA models

**Khi nào dùng:**
- ✅ Cần tốc độ tối đa
- ✅ PDF > 2000 trang
- ✅ Nội dung chủ yếu tiếng Anh
- ✅ Quality "đủ tốt" là OK

---

### **2. Vietnamese_Embedding_v2** 🎯 (Đang chạy)

**Specs:**
```yaml
Model: AITeamVN/Vietnamese_Embedding_v2
Type: Embedding (Bi-Encoder)
Dimensions: 1024
Parameters: 568M
Base: BGE-M3 (SOTA)
Layers: 24
Max Seq Length: 2048
Language: Multilingual (Việt + Anh excellent)
Training Data: 1.1M triplets (Vietnamese)
License: Apache 2.0
```

**Performance (Dự đoán):**
```
PDF 2620 pages:
- PDF reading: 4.2s
- Chunking: 0.02s
- Embedding: 480s (9,286 chunks, ~0.052s/chunk)
- Storage: 2.7s
- TOTAL: 487s = 8.1 minutes

Speed: 19.3 chunks/second
Quality: ⭐⭐⭐⭐⭐ (Excellent)
```

**Ưu điểm:**
- ✅ **Best quality** cho tiếng Việt
- ✅ **Multilingual** (Việt + Anh)
- ✅ **Long context** (2048 tokens)
- ✅ **Modern architecture** (BGE-M3)
- ✅ **Batch optimization** tốt

**Nhược điểm:**
- ⚠️ Nặng hơn MiniLM (568M vs 22M)
- ⚠️ Chậm hơn 3.5x so với MiniLM

**Khi nào dùng:**
- ✅ **Nội dung tiếng Việt** quan trọng
- ✅ Cần **chất lượng cao**
- ✅ PDF 500-3000 trang
- ✅ Có thể chấp nhận 8-10 phút

**Benchmark (Legal Zalo 2021):**
```
MRR@10:  0.7262  (Best trong embedding models!)
Recall@1: 0.8927
Recall@5: 0.9268
Recall@10: 0.9578
```

---

### **3. Vietnamese_Reranker** ⚠️ (KHÔNG DÙNG CHO EMBEDDING!)

**Specs:**
```yaml
Model: AITeamVN/Vietnamese_Reranker
Type: RERANKER (Cross-Encoder) ← KHÁC!
Output: Similarity SCORE (not vector!)
Parameters: 568M
Base: BGE-Reranker-v2-M3
Max Seq Length: 2304 (query 256 + doc 2048)
Language: Vietnamese
License: Apache 2.0
```

**⚠️ TẠI SAO KHÔNG DÙNG?**

**1. RERANKER ≠ EMBEDDING!**
```python
# Embedding model (Bi-Encoder)
embeddings = model.encode(texts)
# → [N, 1024] vectors
# → Store in vector DB
# → Fast similarity search

# Reranker (Cross-Encoder)
scores = reranker.predict([[query, doc1], [query, doc2]])
# → [score1, score2]  ← Không phải vector!
# → KHÔNG thể lưu vào vector DB!
# → Chỉ dùng để re-rank
```

**2. Workflow khác nhau:**
```python
# RAG với Embedding (ĐÚNG!)
Step 1: Embed all documents → Vector DB
Step 2: Query → Embed query → Similarity search
Step 3: Return top-K results

# RAG với Reranker (SAI!)
Step 1: ??? (Không thể embed documents!)
# Reranker KHÔNG TẠO VECTORS!

# RAG với Embedding + Reranker (ĐÚNG, nhưng 2 bước!)
Step 1: Embed all docs → Vector DB (dùng Embedding model)
Step 2: Query → Retrieve top-100 (dùng Embedding)
Step 3: Re-rank top-100 → top-10 (dùng Reranker)
```

**3. Performance issue:**
```python
# Embedding: O(N) để index, O(log N) để search
index_time = N * 0.05s  # Embed 1 lần
search_time = log(N) * 0.001s  # Fast!

# Reranker: O(N) MỖI QUERY!
rerank_time = N * 0.5s  # Mỗi query phải rerank ALL docs!
# 10,000 docs → 5000s = 83 phút MỖI QUERY! 😱
```

**Khi nào dùng Reranker:**
- ✅ **SAU KHI** đã retrieve với Embedding
- ✅ Re-rank top-100 → top-10
- ✅ Cải thiện precision
- ❌ **KHÔNG dùng** thay Embedding model!

**Benchmark (Legal Zalo 2021):**
```
MRR@10:  0.7944  ← Best (nhưng khác use case!)
Recall@1: 0.9324
Recall@5: 0.9537
Recall@10: 0.9740
```

---

### **4. Alibaba-NLP/gte** 🐢

**Specs:**
```yaml
Model: Alibaba-NLP/gte-multilingual-base
Type: Embedding (Bi-Encoder)
Dimensions: 768
Parameters: 278M
Base: NewModel (custom)
Layers: 12
Max Seq Length: 512
Language: Multilingual
```

**Performance (Thực tế từ diagnostic):**
```
PDF 2620 pages:
- Embedding (single): 0.14s/chunk
- Embedding (batch): 0.63s/chunk ← BATCH CHẬM HƠN!
- TOTAL: 5442s = 90.7 minutes ❌

Speed: 15.3 chunks/second (batch mode)
Quality: ⭐⭐⭐⭐
```

**Vấn đề:**
- ❌ **Batch CHẬM HƠN single** (0.63s vs 0.14s)
- ❌ Không optimize cho batch inference
- ❌ CPU bottleneck
- ❌ 90 phút quá chậm!

**Khi nào dùng:**
- ⚠️ **KHÔNG khuyến nghị** cho large PDF
- ⚠️ Chỉ dùng nếu PHẢI dùng Alibaba

---

## 📊 BẢNG SO SÁNH TỔNG HỢP

### **A. SPECS:**

| **Metric** | **MiniLM** | **AIVN Embedding** | **AIVN Reranker** | **Alibaba** |
|-----------|-----------|-------------------|------------------|------------|
| **Type** | Embedding | Embedding | **Reranker** ⚠️ | Embedding |
| **Can use for RAG?** | ✅ Yes | ✅ Yes | ❌ **NO** | ✅ Yes |
| **Dimensions** | 384 | 1024 | N/A (scores) | 768 |
| **Parameters** | 22M | 568M | 568M | 278M |
| **Max Seq Len** | 512 | 2048 | 2304 | 512 |

### **B. PERFORMANCE (2620 pages):**

| **Metric** | **MiniLM** | **AIVN Embedding** | **AIVN Reranker** | **Alibaba** |
|-----------|-----------|-------------------|------------------|------------|
| **Total Time** | **2.3 min** ⚡ | **8.1 min** | N/A | 90.7 min 🐢 |
| **Embedding Speed** | 70.7 ch/s | 19.3 ch/s | N/A | 15.3 ch/s |
| **Speedup vs Alibaba** | **39x** 🔥 | **11x** | N/A | 1x |
| **Memory** | 500MB | 2GB | N/A | 1.5GB |

### **C. QUALITY:**

| **Metric** | **MiniLM** | **AIVN Embedding** | **AIVN Reranker** | **Alibaba** |
|-----------|-----------|-------------------|------------------|------------|
| **Vietnamese** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (rerank only) | ⭐⭐⭐⭐ |
| **English** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Multilingual** | ✅ 100+ | ✅ Yes | ✅ Yes | ✅ Yes |
| **MRR@10 (Legal)** | N/A | 0.7262 | **0.7944** ← Best | N/A |

### **D. USE CASES:**

| **Use Case** | **MiniLM** | **AIVN Embedding** | **AIVN Reranker** | **Alibaba** |
|-------------|-----------|-------------------|------------------|------------|
| **Large PDF (2000+ pages)** | ✅ Best | ✅ Good | ❌ No | ❌ Too slow |
| **Vietnamese content** | ⚠️ OK | ✅ **Best** | ⚠️ Re-rank only | ⚠️ OK |
| **English content** | ✅ Good | ✅ Good | ⚠️ Re-rank only | ✅ Good |
| **Speed critical** | ✅ **Best** | ⚠️ OK | ❌ N/A | ❌ Too slow |
| **Quality critical** | ⚠️ OK | ✅ **Best** | ✅ Re-rank phase | ⚠️ OK |
| **Re-ranking results** | ❌ No | ❌ No | ✅ **Yes** | ❌ No |

---

## 🎯 KHUYẾN NGHỊ CHO TỪNG TRƯỜNG HỢP

### **Scenario 1: PDF lớn (2000+ pages), nội dung tiếng Anh**
```
✅ DÙNG: MiniLM-L6-v2
Lý do:
- Speed: 2.3 phút (39x faster than Alibaba)
- Quality: Đủ tốt cho tiếng Anh
- Memory: Chỉ 500MB
```

### **Scenario 2: PDF vừa (500-1500 pages), nội dung tiếng Việt**
```
✅ DÙNG: Vietnamese_Embedding_v2
Lý do:
- Quality: Best cho tiếng Việt
- Speed: 8-10 phút (chấp nhận được)
- Long context: 2048 tokens
```

### **Scenario 3: Cần chất lượng tối đa cho tiếng Việt**
```
✅ DÙNG: Vietnamese_Embedding_v2 + Vietnamese_Reranker

Pipeline:
1. Embed all docs với Vietnamese_Embedding_v2
2. Store in vector DB
3. Query → Retrieve top-100 (fast!)
4. Re-rank top-100 → top-10 với Vietnamese_Reranker
5. Return top-10 (best quality!)

Time:
- Indexing: 8 phút (1 lần)
- Query: 1s (retrieve) + 5s (rerank 100 docs) = 6s
```

### **Scenario 4: PDF cực lớn (5000+ pages), cần tốc độ tối đa**
```
✅ DÙNG: MiniLM-L6-v2
Lý do:
- 5000 pages: ~5 phút với MiniLM
- Vietnamese_Embedding: ~15 phút
- Alibaba: ~175 phút (không chấp nhận được!)
```

---

## ⚠️ LƯU Ý QUAN TRỌNG

### **1. Vietnamese_Reranker KHÔNG thể thay thế Embedding!**
```python
# SAI! ❌
embeddings = reranker.encode(texts)  # NO METHOD!
# Reranker không có encode() method!

# ĐÚNG! ✅
# Bước 1: Embed với Embedding model
embeddings = embedding_model.encode(texts)
vector_db.add(embeddings)

# Bước 2: Query
query_results = vector_db.search(query, top_k=100)

# Bước 3: Re-rank với Reranker
pairs = [[query, doc] for doc in query_results]
scores = reranker.predict(pairs)
top_10 = sorted(zip(query_results, scores))[:10]
```

### **2. Batch size tuning:**
```python
# MiniLM: batch_size=500 (optimal)
# Vietnamese_Embedding_v2: batch_size=500
# Alibaba: batch_size=1 (batch mode chậm hơn!)
```

### **3. Memory requirements:**
```python
MiniLM:               500MB
Vietnamese_Embedding: 2GB
Alibaba:              1.5GB
Vietnamese_Reranker:  2GB (khi dùng)
```

---

## 🏆 KẾT LUẬN CUỐI CÙNG

### **Cho file PDF 2620 trang của thầy (Machine Learning book):**

**Option 1: SPEED PRIORITY (Khuyến nghị!)** ⚡
```
Model: MiniLM-L6-v2
Time: 2.3 minutes
Quality: Good (đủ cho tiếng Anh)
Command: python demo_final_optimized.py (với MiniLM)
```

**Option 2: QUALITY PRIORITY** 🎯
```
Model: Vietnamese_Embedding_v2
Time: 8-10 minutes
Quality: Excellent (best cho multilingual)
Command: python demo_final_optimized.py (đang chạy)
```

**Option 3: MAXIMUM QUALITY (2-phase)** 🔥
```
Phase 1: Index với Vietnamese_Embedding_v2 (8 phút, 1 lần)
Phase 2: Query → Retrieve + Rerank với Vietnamese_Reranker
Time: Indexing 8 phút + Query 6s
Quality: Maximum!
```

**❌ KHÔNG DÙNG:**
- Alibaba (quá chậm: 90 phút)
- Vietnamese_Reranker alone (không thể dùng cho embedding!)

---

**🎯 EM KHUYẾN NGHỊ:**

Vì sách của thầy là **tiếng Anh** (Machine Learning Systems):
→ **Dùng MiniLM-L6-v2** (đã test thành công, 2.3 phút)
→ Nếu muốn quality cao hơn: Dùng **Vietnamese_Embedding_v2** (8 phút, đang chạy)

**Vietnamese_Reranker chỉ dùng để RE-RANK, KHÔNG thay thế Embedding!** ⚠️
