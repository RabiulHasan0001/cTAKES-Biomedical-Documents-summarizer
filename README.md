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

## 📁 Repository Structure

