import streamlit as sl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
api_key = ''
def load_knowledgeBase(api_key):
        embeddings=OpenAIEmbeddings(api_key=api_key)
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db

def load_prompt():
        prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the pdf , answer "i donot know what the hell you are asking about"
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt

#to load the OPENAI LLM
def load_llm(api_key):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)
        return llm

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
sl.header("welcome to the pdf bot")
knowledgeBase=load_knowledgeBase(api_key)
llm=load_llm(api_key)
prompt=load_prompt()

query=sl.text_input('Enter some text')


if(query):
        #getting only the chunks that are similar to the query for llm to produce the output
        similar_embeddings=knowledgeBase.similarity_search(query)
        similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key=api_key))
        
        #creating the chain for integrating llm,prompt,stroutputparser
        retriever = similar_embeddings.as_retriever()
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        
        response=rag_chain.invoke(query)
        sl.write(response)
