# Bug-triage-project
AI-Powered Bug Triage System
An end-to-end Machine Learning project designed to automate the classification of technical GitHub issues. This system bridges the gap between traditional statistical NLP and modern Generative AI.


🚀 The Problem
Open-source maintainers spend hundreds of hours manually triaging incoming issues. Existing automated systems often rely on simple keyword matching, which fails to capture the technical intent of complex bug reports.


🛠 Project Pipeline
Fault-Tolerant Scraping: Developed a custom Python scraper with a built-in retry mechanism to extract structured data from large GitHub repositories while respecting API rate limits.
Statistical Baseline: Established a  benchmark accuracy using Multinomial Naive Bayes and TF-IDF vectorization to quantify the necessity of more advanced models.
LLM Fine-Tuning: Leveraged Llama-3-8B using LoRA (Low-Rank Adaptation) to achieve semantic classification.
Optimization: Implemented 4-bit quantization via bitsandbytes to perform fine-tuning on resource-constrained hardware (Google Colab T4 GPU).
