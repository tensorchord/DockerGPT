import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()

if 'OPENAI_API_KEY' not in os.environ:
    print("Please set 'OPENAI_API_KEY' to system env first.")
    quit(1)

root_dir = '../../aiac'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    print(dirpath, filenames)
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

# TODO: Try various arguments and find out their influence to final result
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 20

model = ChatOpenAI(model='gpt-3.5-turbo')
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

pre_questions = []
chat_history = []

print("Start chatting, enter 'quit' to end")
while True:
    question = input("-> **Question**: ")
    if question.lower() == "quit":
        break
    try:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"**Answer**: {result['answer']} \n")
    except Exception as e:
        print(str(e))
