# 🏛️ Tax & VAT RAG Assistant 🤖💶

> **An intelligent AI assistant powered by a RAG (Retrieval-Augmented Generation) architecture for consulting Italian and European VAT/Tax regulations (OSS, IOSS, E-commerce, and EU Directives). The PDFs are coming from Italian version so the system prompt provided is in EN or IT**

---

## 🎯 Project Overview

The **Tax & VAT RAG Assistant** provides precise, structured, and directly referenced answers regarding the **VAT** regulatory framework (EU Directives, Italian DPR 633/72, special OSS/IOSS schemes, and cross-border e-commerce).

Built on a **serverless RAG architecture** hosted on [Modal.com](https://modal.com):
* 📚 **Indexes and queries** a curated set of regulatory PDF documents.
* 🔍 **Executes high-precision semantic searches** using `ChromaDB` and multilingual embedding models (`bge-m3`).
* ⚡ **Generates fast, context-aware responses** via high-performance LLMs served through **Groq** APIs.
* 🌐 **Provides a modern web interface** (FastAPI + HTML5/CSS3) secured by Basic Authentication and featuring chat history persistence saved across persistent volumes.

---

## 📚 Indexed Document Set

The document corpus covers the evolution of both national and European VAT law, with a dedicated focus on e-commerce regimes and cross-border transactions:

| 📄 Document / File | 🛡️ Regulatory Scope & Purpose |
| :--- | :--- |
| `CELEX_02006L0112-20240101_IT_TXT.pdf` | **Directive 2006/112/EC (Consolidated 2024)**<br>The foundational text of the common system of value added tax in the European Union. |
| `DPr 633_1972_DPr 633_1972.pdf` | **DPR October 26, 1972, n. 633**<br>Italian Presidential Decree governing the establishment and management of VAT in Italy. |
| `Direttiva_UE_2017_2455_Ecommerce.pdf` | **EU Directive 2017/2455**<br>Amendments to VAT directives for provision of services and distance sales of goods (E-commerce Package). |
| `Explanatory Notes_revised_1Jan2027_0.pdf` | **EU Explanatory Notes (E-commerce / ViDA)**<br>European Commission explanatory notes on updated e-commerce rules and perspectives through 2027. |
| `OSS Guidelines_revised_1Jan2027_0.pdf` | **EU OSS/IOSS Guidelines**<br>Practical guidance on applying the One Stop Shop (OSS) and Import One Stop Shop (IOSS) special schemes for e-commerce. |
| `Provvedimento uffici regime Oss Ioss 25.06.2021 (1).pdf` | **Italian Revenue Agency Provision (25/06/2021)**<br>Implementation rules for registering and managing OSS and IOSS schemes in Italy. |
| `Regolamento_Esecuzione_UE_282_2011.pdf` | **Implementing Regulation (EU) No 282/2011**<br>Measures for applying Directive 2006/112/EC on place of supply, territoriality, and digital services. |

---

## 📁 Project Structure & Folder Analysis

Overview of the project directory structure across local and remote environments:

```text
.
├── 📄 app.py                                   # 🚀 Core app: Modal service + FastAPI backend + UI router
├── 📄 index.html                               # 🎨 Web UI Frontend
├── 📄 preprocess_and_chuk.py                   # ✂️ PDF text extraction & chunking script
├── 📄 ingest_to_chroma.py                      # 📥 Vector ingestion script into ChromaDB
├── 📄 test_rag.py                              # 🧪 RAG pipeline benchmark script
├── 📁 documenti_iva_oss/                       # 📂 Source PDF files (used for ingestion)
├── 📁 chroma_db_iva/                           # 🗄️ Persistent ChromaDB vector store
├── 📁 storage/                                 # ⚠️ [REMOVE - Deprecated or residual test folder]
├── 📁 temp/                                    # ⚠️ [REMOVE - Temporary processing folder]
└── 📁 __pycache__/                             # 🧹 Python bytecode cache (auto-generated)