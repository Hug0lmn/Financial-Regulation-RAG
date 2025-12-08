# Financial Regulation RAG

This repository investigates the application of Retrieval-Augmented Generation (RAG) to banking and financial regulatory texts.  
The goal is to build a clean, reproducible baseline pipeline that can later be expanded with more advanced retrieval, preprocessing, and prompting techniques.  

Current focus: **IFRS 7/9/13** as the initial regulatory corpus and english corpus.  

---  

## Notes  

Regarding RAG (not yet organized): https://www.notion.so/RAG-informations-advices-2b111b4b8f9c80d48216fd05e8aefd92?source=copy_link  

## Current Capabilities (Baseline)  

### 1. Preprocessing & Cleaning  
- Automatic preprocessing of IFRS template documents.  
- Early restructuring attempts to better align sections, subparagraphs, lists, and definitions.  

### 2. Chunking & Retrieval  
- Basic chunking logic.  
- Construction of a simple vector store from cleaned text.  
- Basic metadata handling (section ID, sub-paragraph markers).  

### 3. Model Interaction  
- Minimal prompting pipeline: retrieved context → answer generation.  
- Currently operates using a small, free LLM due to hardware constraints.  
- Acts as a **baseline** rather than a fully optimized RAG system.  

This baseline exists so that future improvements can be evaluated against a stable reference point.  

---  

## Limitations (Intentional at This Stage)  

- Slow generation (dependant on hardware). 
- Prompting strategy not optimal and result not sufficient.  

These limitations form the basis of the roadmap for future work.  

---  

## Next Steps  

### 1. Expand and Diversify the Regulatory Corpus  
- Add Basel II/III/IV, EBA Guidelines, and local regulatory frameworks.  
- Improve the cleaning pipeline with modular components and specialized structure.  

### 2. Develop a Streamlit Interface  
- Simple UI to query the corpus and visualize retrieval results.  
- Display retrieved chunks, metadata, and the model’s reasoning steps.  
- Useful for comparing improvements across versions.  

### 3. Enhance Retrieval Quality  
- Test modern embedding models.  
- Explore other chunking methods.  
- ...  

### 4. Strengthen Prompting & Context Integration  
- Structured templates for regulatory explanations.  
- Retrieval filtering, re-ranking, etc...  
- Context window optimization for long, hierarchical regulations.  

### 5. Try implemented a RAG Graph  

---
