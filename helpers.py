import os
# import uuid
import time
import requests
import asyncio
import openai
# import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback

ALL_USER_FILES = {}
MAX_ALLOWED_TIME_SEC = 900
MAX_ALLOWED_USERS = 10

def setup_api_key():
    openai.api_key = os.environ["OPENAI_API_KEY"]

def log_dirs(path='/kaggle/working'):
    print(f"<<<Directory Tree For {path}:>>>", end="")
    for pwd, dirs, filenames in os.walk(path):
        print("\n" + pwd + " :")
        if len(dirs + filenames) > 0:
            print("  " + ", ".join(dirs + filenames))
    print("<<<Tree End>>>")

def download_default_doc(download_url, download_loc):    
    resp = requests.get(download_url)
    with open(download_loc, "wb") as f1:
        f1.write(resp.content)

def load_docs(paths):
    pages = []
    for p in paths:
        loader = PyPDFLoader(p)
        pages += loader.load()
    return pages

def split_docs(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_documents(pages)
    return chunks

def create_vector_db(chunks, persist_directory):
    embedding = OpenAIEmbeddings()
    
    print(f"<<<Removing files under ./{persist_directory}>>>")
    os.system(f"rm -rf ./{persist_directory}")  # remove old database files if any
    print("<<<Remove End>>>")
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    print("VectorDB Collection Count:")
    print(vectordb._collection.count())
    vectordb.persist()
    return vectordb, embedding

def create_retriever(vectordb, embedding):
    retriever = vectordb.as_retriever(search_type="mmr")

    # splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
    # relevant_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, 
        base_retriever=retriever
    )
    
    return compression_retriever

def create_conv_chain(compression_retriever):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qachain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
#         memory=buffer_memory,
        retriever=compression_retriever,
        return_source_documents=True,
        return_generated_question=True,
    #     verbose=True,
    #     callbacks=[MyCustomHandler()]
    )
    return qachain

def create_db_by_loading_docs(paths=["users/0/files/MachineLearning-Lecture01.pdf"],
                              user_id="0"):
    print(f"----- Start Creating DB for User {user_id} -----")
    print("Uploaded Files:", paths)
    has_uploaded = False
    if user_id != "0":
#         await delete_db_after_timeout(user_id)
#         task1 = asyncio.create_task(delete_db_after_timeout(user_id))
#         print("Timer set:", user_id)
        curr_time_sec = time.time()
        remove_prev_n_excess_files(curr_time_sec)
        add_new_user_file(user_id=user_id, upl_time_sec=curr_time_sec)
        show_all_user_files()
        os.system(f"mkdir -p users/{user_id}/chroma")
        log_dirs(f"./users/{user_id}")
        has_uploaded = True
    pages = load_docs(paths)
    chunks = split_docs(pages)
    persist_directory = f"users/{user_id}/chroma/"
    vectordb, embedding = create_vector_db(chunks, persist_directory)
    log_dirs(persist_directory)
    print("<<<Files under ./users/ :>>>")
    os.system("ls users")
    print(f"----- Complete Creating DB -----")
    return has_uploaded

def create_chain(user_id="0", has_uploaded=False):
    print(f"----- Start Creating Chain for User {user_id} -----")
    show_all_user_files()
    print("<<<Files under ./users/ :>>>")
    os.system("ls users")
    
    if user_id != "0" and has_uploaded == True:
        if check_user_file_exists(user_id):
            persist_directory = f"users/{user_id}/chroma/"
        else:
            print("Sorry! Timeout! Your files were deleted! Please re-upload!")
            raise Exception("Sorry! Timeout! Your files were deleted! Please re-upload!")
    else:
        persist_directory = "users/0/chroma/"

    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    compression_retriever = create_retriever(vectordb, embedding)
    qachain = create_conv_chain(compression_retriever)
    print(f"----- Complete Creating Chain -----")
    return qachain

def initial_setup():
    print("----- Start Initial Setup -----")
    setup_api_key()
    
    os.system("mkdir -p users/0/chroma")
    os.system("mkdir -p users/0/files")
    
    log_dirs("./")
    
    download_loc="users/0/files/MachineLearning-Lecture01.pdf"
    download_url = "https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf"
    download_default_doc(download_url, download_loc)
    log_dirs("/".join(download_loc.split("/")[:-1]))
    
    create_db_by_loading_docs()
    print("----- Complete Initial Setup -----")

def convert_hist_ui_to_chat_hist(hist_ui):
    chat_history = []
    for turn in hist_ui:
        usermsg, assmsg = turn
        chat_history += [(usermsg, assmsg)]
#         chat_history += f"""Human: {usermsg}
# Assistant: {assmsg}
# """
    return chat_history

def process_input(user, hist_ui, user_no, has_uploaded):
    if user.strip() == "":
        raise Exception("Empty query!")
    time.sleep(17)
    # qachain.memory.clear()
    # convert_hist_ui_to_mem(hist_ui)
    # with get_openai_callback() as cb:
    qachain = create_chain(user_no, has_uploaded)
    chat_history = convert_hist_ui_to_chat_hist(hist_ui)
    response = qachain({"question": user.strip(), "chat_history": chat_history}) 
    source = [r.to_document() for r in response["source_documents"]]
    source = [( 
            "**Text:** " + r.page_content + "  \n" +
            "**Page:** " + str(r.metadata["page"]) + " | " +
            "**File:** " + r.metadata["source"].split("/")[-1]
            ) for r in source
        ]
    source = [str(r) for r in source]
    source_str = "\n\n".join(source)
    db_query = response["generated_question"]
    # hist_ui = convert_mem_to_hist_ui(qachain)
    # qachain.memory.clear()
    ## History for ui, Total Tokens, Prompt Tokens, Completion Tokens, Total Cost (USD)
    # hist_ui, cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost
    # history + [[user, user]], "query: " + user, "docs: " + user
    return response["answer"], db_query, source_str

def remove_prev_n_excess_files(curr_time_sec):
    max_time_sec = MAX_ALLOWED_TIME_SEC
    max_users = MAX_ALLOWED_USERS
    users = ALL_USER_FILES
    remove_files = []
    excess = -1

#     remove_files += [
#         fil for fil in all_files if curr_time_sec - fil["upl_time_sec"] >= max_time_sec
#     ]
#     all_files = [
#         fil for fil in all_files if curr_time_sec - fil["upl_time_sec"] < max_time_sec
#     ]

    remove_users = [k for k, v in users.items() if curr_time_sec - v >= max_time_sec]
    remain_users = [k for k, v in users.items() if curr_time_sec - v < max_time_sec]
    
    n = len(remain_users)
    
    if n > max_users:
        excess = (n - max_users) + 1
    elif n == max_users:
        excess = 1

    if excess > 0:
        for i in range(excess):
            remove_users.append(remain_users[i])
    
    for uid in remove_users:
        del ALL_USER_FILES[uid]
        os.system(f"rm -rf users/{uid}")
        print(f"<<Deleted {uid}>>")
    
    del remain_users
    del remove_users

def add_new_user_file(user_id, upl_time_sec):
    ALL_USER_FILES[user_id] = upl_time_sec

# def set_all_user_files(all_files):
#     n = len(ALL_USER_FILES)
#     for i in range(n):
#         ALL_USER_FILES.pop()
#     for i, e in enumerate(all_files):
#         ALL_USER_FILES.append(e)
    
def show_all_user_files():
    print("<<<All user ids:>>>")
    print(ALL_USER_FILES)

def check_user_file_exists(user_id):
    return user_id in ALL_USER_FILES