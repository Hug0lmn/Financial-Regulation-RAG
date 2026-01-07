import ui as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retriever.final_retriever import production_retriever
from rag.chain import create_rag_chain
from LLM import llm

model = llm.import_llm("qwen2.5-0.5b-instruct-q8_0.gguf")
retrieve_hybrid = production_retriever()
chain = create_rag_chain(model, retriever=retrieve_hybrid, include_sources=False)

st.title("Mini RAG (IFRS / Réglementation)")

query = st.text_input("Entrez votre question")
query_prefix = "query :"
full_query = query_prefix+query

if query:
    with st.spinner("Recherche en cours…"):
        result = chain.invoke(full_query)

    st.subheader("Réponse")
    st.write(result["answer"])

    st.subheader("Sources")
    list_sources = [elem.metadata for elem in result["retrieved_documents"]]
    all_list = [
        elem.get("source", "") + " - " + elem.get("title", "") + " - " + elem.get("subtitle", "") + " - " + elem.get("subsection", "")
        for elem in list_sources]

    for i, label in enumerate(all_list):
        if label in all_list[:i]:
            continue
        with st.expander(label):
            st.write("Contenu :")
            content = "\n\n".join(
                [elem.page_content for elem in result["retrieved_documents"] if
                 elem.metadata.get("source", "") + " - " + elem.metadata.get("title", "") + " - " +
                 elem.metadata.get("subtitle", "") + " - " + elem.metadata.get("subsection", "") == label])
            st.write(content)
