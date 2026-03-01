# Endee Powered OCR RAG Chatbot

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system using OCR and Endee as the vector database.

## Problem Statement
Large Language Models hallucinate when they lack context. This system retrieves semantically relevant document chunks using vector similarity search before generating answers.

## System Architecture
Upload Image → OCR → Chunking → Embedding → Endee Vector DB → Similarity Search → LLM

## How Endee is Used
- Stores document embeddings
- Performs vector similarity search
- Retrieves top-k relevant chunks
- Enables semantic document question answering

## Repository Compliance
Forked Endee Repository:
https://github.com/Arpitkochar1/endee

## Setup Instructions

1. Star official Endee repository
2. Fork official Endee repository
3. Clone your fork
4. Install Endee locally:
   pip install -e .

5. Install project dependencies:
   pip install -r requirements.txt

6. Run:
   uvicorn app:app --reload
