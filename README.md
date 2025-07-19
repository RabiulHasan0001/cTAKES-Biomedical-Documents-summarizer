# cTAKES-Biomedical-Documents-summarizer

A **clinical NLP toolkit** built for ontology-based and semantic-aware summarization of PubMed **biomedical articles** and **clinical case studies**. This system leverages medical concept extraction (via **cTAKES**) and semantic chaining (via **SpaCy vectors**) to produce interpretable summaries using **LexRank**.

This repository includes:
- A full **Tkinter-based GUI** for medical text exploration, parsing, entity recognition, and summarization
- Tools to analyze both **structured biomedical literature** and **narrative clinical notes**

---

##  Use Case

Designed for:
- NLP researchers working with **PubMed abstracts**, **PMC full-texts**, and **case study corpora**
- Medical informaticians needing interpretable summarization without neural models
- Developers integrating biomedical NLP with clinical entity frameworks like **UMLS**

---

## 💡 Features

###  Ontology-Driven Summarization Pipeline
- Extracts relevant sentences using:
  - **cTAKES**: biomedical entities (Diseases, Procedures, Medications)
  - **SpaCy similarity chains**: selects semantically related sentences
- Summarization via **LexRank** applied to semantically ranked + concept-aware sentences
- Built-in tokenizer and filter for PubMed-style content

###  GUI (Tkinter): Clinical NLP Dashboard
- Load case reports or PubMed papers from file or web
- Run full **cTAKES Clinical Pipeline**
- Display parsed output, negation, polarity, and SRL
- Generate **summaries with biomedical context**
- Highlighted named entities in multi-color viewer

---

## 🧰 NLP Toolkit

###  General NLP Processing

| Module | Description |
|--------|-------------|
| General NER | Extracts PERSON, ORG, DATE using spaCy |
| Chunk Parsing | Extracts NP, VP, PP, CLAUSE patterns via NLTK |
| Constituency Parsing | Stanford Parser integration for visual phrase structure |
| Dependency Parsing | Uses Displacy to generate interactive SVG graphs |
| Semantic Role Labeling | Adds SRL tags using cTAKES predicate frames |

###  Medical NLP Processing

| Module | Description |
|--------|-------------|
| Clinical NER | Extracts key concepts using cTAKES XMI: Diseases, Drugs, Symptoms |
| Polarity Analysis | Highlights positive/negative assertions |
| Clinical Negation | Detects negated biomedical mentions via `negspacy` |
| Ontology-Guided Summarization | LexRank applied to concept-aware sentences with semantic chains |

---

📂 gui/

└── medical_text_ontology_analyzer.py    # GUI-based clinical NLP tool

📂 chaining/

└── lexical_chains_spacy.py              # Sentence similarity using SpaCy vectors

📂 evaluation/

└── summarization_evaluation.py          # CLI tool for summarization metrics & visualization

📂 config/

└── config.yaml                          # Configuration file for external tools 



---

## ⚙️ Installation

### 🔹 Dependencies

```bash
pip install pandas spacy nltk stanza spacy-stanza xmltodict \
  beautifulsoup4 requests matplotlib tqdm scikit-learn \
  sentence-transformers bert-score rouge-score sumy negspacy pyyaml

python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
python -m stanza download en --package=mimic --processors={'ner': 'i2b2'} 
```

---
## 🛠️ External Setup

### 🔸 Stanford Parser

**Download from:**  
[https://stanfordnlp.github.io/CoreNLP/download.html](https://stanfordnlp.github.io/CoreNLP/download.html)

**Update `config.yaml`:**

```yaml
stanford_parser:
  path_to_jar: "/path/to/stanford-parser.jar"
  path_to_models_jar: "/path/to/stanford-parser-4.x.x-models.jar"
```

### 🔸 Apache cTAKES

Download from:  
https://ctakes.apache.org/downloads.html
**Update `config.yaml`:**

```yaml
ctakes:
  installation_dir: "/path/to/ctakes"
  input_dir: "/path/to/ctakes/input"
  output_dir: "/path/to/ctakes/output"
  pipeline_key: "your_UMLS_API_key"
```


## 🖥️ How to Use

### ✅ GUI (Recommended)

```bash
# Run GUI application
python gui/medical_text_ontology_analyzer.py

# Steps:
# - Load PubMed text or Case Study document
# - Run entity extraction via cTAKES
# - Visualize linguistic parses (chunking, dependency, SRL)
# - Click "Summarize" to generate extractive summaries
```

