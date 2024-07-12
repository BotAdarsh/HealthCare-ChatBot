import streamlit as st
import random
import time
from datasets import load_dataset
import spacy
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from qdrant_client import QdrantClient
import numpy as np



qdrant_client = QdrantClient(
    url="https://6c3638cb-0cc2-4378-9aec-3dc127340b1d.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="6T1W-42fAxBxxEM9nq1FLqMqGmBPDQ_UU0yT11KGWMyKv1AqjDIuhg"
)

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def vectorize_query(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def search_in_qdrant(query, tokenizer, model, top_k=10):
    query_vector = vectorize_query(query, tokenizer, model)

    # Search in Qdrant
    hits=[]
    hits.append( qdrant_client.search(
        collection_name="semantic_search_medical_qa",
        query_vector=("qvect",query_vector.tolist()),
        limit=top_k,
    ))
    hits.append(qdrant_client.search(
        collection_name="semantic_search_medical_qa",
        query_vector=("cvect",query_vector.tolist()),
        limit=top_k,
    ))
    hits.append(qdrant_client.search(
        collection_name="semantic_search_medical_qa",
        query_vector=("avect",query_vector.tolist()),
        limit=top_k,
    ))
    return hits

def display_search_results(test_query, tokenizer, model, cutoff_score):
    # print('here')
    hits = search_in_qdrant(test_query, tokenizer, model)
    # print(results)
    final_res=[]
    for results in hits:
      for result in results:
         if result.score >= cutoff_score:
            final_res.append(result)
    final_res.sort(key=lambda x:x.score, reverse=True)
    return final_res
    # for result in final_res:
    #         print("Answer:", result.payload["answer"])
    #         print("Context:", result.payload["context"])
    #         #print("Question:", result.payload["question"])
    #         print("Score:", result.score)
    #         print("-----------")

# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("Medical QA")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a medical question?"):
    questions=[]
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    gen_response=""
    responses = display_search_results(prompt,tokenizer,model,0.9)
    if len(responses)==0:
        gen_response="Sorry 😢, I dont have answer to your question!"
    else:
        gen_response=f"**{responses[0].payload['answer']}** \n\n {responses[0].payload['context']}"

    questions=[x.payload["question"] for x in responses[1:4]]
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(gen_response))
    if len(questions)>0:
        with st.chat_message("assistant"):
            st.markdown("**Suggested Questions**")
            for i in range(len(questions)):
                st.markdown(questions[i])
                
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
