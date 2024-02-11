# Chatbot app using GPT with Retrieval Augmented Generation

This repository contains a basic adaptable Python application designed to work with custom data & OpenAI's GPT models.
Utilizing the LangChain library for LLM access & processing PDF files and, Streamlit for creating a self-hosted web app 
chat interface, this project enables enhanced interaction with AI models by supporting the retrieval-augmented generation
concept.

Retrieval Augmented Generation (RAG) is a technique that combines the generative capabilities of models like GPT with 
a retrieval component. This approach allows the model to pull in relevant information from a dataset (in this case, PDF 
documents) at runtime, leading to outputs that are informed by a broader context than the model's pre-trained knowledge.


## Features

- **Custom Data Processing**: Process PDF files to use as custom data for GPT model inference.
- **Retrieval-Augmented Generation**: Enhance GPT model responses by integrating retrieval from custom data sources.
- **Streamlit Web App**: Interact with the OpenAI's GPT model through a user-friendly chat interface hosted on your own server.
- **OpenAI API Integration**: Seamlessly use OpenAI's API with your own API key for model access.

## Prerequisites

- Python 3.10 or later.
- An active OpenAI API key.

## Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/ASHISRAVINDRAN/custom-chatbot-app.git
cd custom-chatbot-app
pip install -r requirements.txt
```

## Usage

First put any of your data in PDF format in the `data` folder. Update your OPENAI API key in the `secretkey.py` file.
Then, run the Streamlit web app using the command:
```bash
streamlit run chatbot.py
```

## Note
Currently, the code uses the [MITK](https://www.mitk.org)'s documentation as custom data for demonstration purposes.

## Disclaimer
This project is only a proof-of-concept for experimental purpose. It could lack a lot of features including prompt 
engineering and might have bugs. 

**Feel free to fork &/or contribute to this repo.**
