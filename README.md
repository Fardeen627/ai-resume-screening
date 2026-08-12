# AI Resume Screening & RAG Pipeline

## Overview

An AI-powered resume screening system designed to assist recruiters and hiring managers in analyzing candidate resumes against job descriptions.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant candidate profiles and provide context-aware responses for tasks such as candidate matching, comparison, summarization, and analysis.

The pipeline combines semantic retrieval with LLM-based generation to handle the unstructured and varied language found in resumes and job descriptions.

## How It Works

The system follows two main stages:

### 1. Adaptive Retrieval

The retrieval system identifies relevant resumes based on the user's query.

* **Similarity-based retrieval:** When a job description is provided, the system searches the vector database for resumes that are semantically similar to the requirements.
* **Keyword-based retrieval:** Candidate information can also be retrieved directly when specific applicant IDs or information are provided.

### 2. Generation

The retrieved resumes are provided as context to the LLM.

The model can then perform tasks such as:

* Candidate matching
* Resume comparison
* Candidate summarization
* Skill analysis
* Job-description analysis
* Recruitment-related question answering

## Why Resume Screening?

Traditional resume screening systems often rely heavily on keyword matching. This can miss relevant candidates when equivalent skills or experience are expressed using different terminology.

This project explores how semantic retrieval and LLMs can be used to understand the broader context of resumes and job descriptions rather than relying exclusively on exact keyword matches.

## Why RAG?

Retrieval-Augmented Generation allows the LLM to access information from an external knowledge base instead of relying only on information contained within its training data.

In this project, the knowledge base consists of candidate resumes. Relevant resumes are retrieved based on the user's query and supplied to the LLM as context.

This allows the system to generate responses grounded in the retrieved candidate information.

## RAG Fusion

For complex job descriptions, a single search query may not capture all relevant requirements.

RAG Fusion addresses this by generating multiple search queries from the original input, retrieving documents for each query, and combining and reranking the results.

This can improve retrieval quality when a job description contains multiple skills, technologies, responsibilities, or requirements.

## Demo

Demo: Coming soon

Default synthetic resume dataset used for testing:

[synthetic-resumes.csv](data/main-data/synthetic-resumes.csv)

Source job description dataset:

[Kaggle – Jobs and Job Description Dataset](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description)

## System Architecture

### Chatbot

The chatbot incorporates several techniques to improve the retrieval and generation process:

* **Chat history:** Previous conversation context is provided to the LLM to support follow-up questions.
* **Query classification:** The system determines whether the current query requires document retrieval.
* **Adaptive retrieval:** Retrieval is performed only when relevant candidate information is required.
* **Small-to-big retrieval:** The system initially retrieves relevant text chunks and then traces them back to their original documents before providing the complete resume context to the LLM.

### RAG Pipeline

The pipeline processes resumes into vector representations and stores them in a FAISS vector index.

When a job description is provided:

1. The input is processed by the LLM.
2. Multiple search queries can be generated for complex inputs.
3. Each query is used to retrieve relevant resume chunks.
4. Retrieved results are combined and reranked.
5. The most relevant candidate documents are selected.
6. The retrieved resumes are provided to the LLM as context.
7. The LLM generates a response based on the retrieved candidate information.

## Tech Stack

* `Python` — Core implementation
* `LangChain` — RAG pipeline and LLM integration
* `OpenAI` — LLM integration
* `Hugging Face` — Embedding models
* `FAISS` — Vector indexing and similarity search
* `Streamlit` — User interface

## Installation

Clone the repository:

```bash
git clone https://github.com/Fardeen627/ai-resume-screening.git
cd ai-resume-screening
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run demo/interface.py
```

## Data

The repository contains datasets used for development and evaluation.

The synthetic resume dataset is included in:

```text
data/main-data/synthetic-resumes.csv
```

The job description dataset is sourced from Kaggle.

## Project Status

This project is currently under development.

Future improvements may include:

* Improved candidate ranking
* Skill-gap analysis
* Better retrieval evaluation
* Additional embedding models
* Resume parsing improvements
* Candidate scoring
* API-based backend
* Production deployment
* Automated evaluation

## Contributions

Suggestions and improvements are welcome.

Please open an issue in this repository if you encounter a bug or have an idea for improving the project.

## Acknowledgements

This project was developed using concepts and techniques from the broader RAG ecosystem.

Special acknowledgement to the following resources:

* [RAG Fusion](https://github.com/Raudaschl/rag-fusion) — inspiration for the multi-query retrieval and reranking approach.
* [Kaggle Jobs and Job Description Dataset](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description) — source of job description data.

The project is an adaptation and further development of an existing resume-screening RAG implementation. Appropriate credit is retained to the original resources and datasets used.
