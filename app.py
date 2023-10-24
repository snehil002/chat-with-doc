import uuid
import gradio as gr
from helpers import process_input, create_db_by_loading_docs, initial_setup


initial_setup()



##### UI Functions ####

def format_file_names(file_paths):
    files_str = "\n".join(["* " + path for path in file_paths])
    return f"""**Currently Loaded Files:**
{files_str}"""


# def format_current_query(query):
#     return f"""**Current DB Query:**  
# {query}"""


# def format_retrieved_docs(docs):
#     return f"""**Retrieved Documents:**  
# {docs}"""


# def format_info(user_state, docs_state, upl_state):
#     return f"""**User ID:**  
# {user_state}  
# {format_file_names(docs_state)}  
# **Has Uploaded:**  
# {upl_state}"""


def generate_unique_no():
    uid = uuid.uuid4()
    print("Generated User Id:", uid)
    return str(uid)


def ui_func(user, history):
    return "", history + [[user, None]]


def ui_func_2(history, user_state, upl_state, log_state):
#     print("User ID:", user_state)
    user = history.pop()[0]
    curr_query = ""
    retrieved_docs = ""
    
    try:
        response, curr_log = process_input(
            user, history, user_state, upl_state
        )
        history += [[user, response]]
        user = ""
        log_state += [curr_log]
    
    except Exception as exp:
        print("Error!")
        print(exp)
        gr.Error(str(exp))
    
    return user, history, log_state, log_state


def show_files_and_create_chain_ui(files, user_state, docs_state, upl_state, curr_files):
    try:
        gr.Info("Your Files Are Uploaded!")

        file_paths = [file.name for file in files]
        upl_state = create_db_by_loading_docs(file_paths, user_state)
        
        file_paths = [file.split("/")[-1] for file in file_paths]
        docs_state = file_paths
        curr_files = format_file_names(docs_state)

        gr.Info("Your Files Are Now Ready For QnA!")
    
    except Exception as exp:
        print("Error!")
        print(exp)
        gr.Error(str(exp))
    
    return docs_state, upl_state, curr_files



####### UI Starts here #######

css="""
.upl-but{ height:50px; min-height: 150px }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("""# PDF Sage: Intelligent Document Query Chatbot""")
    
    docs_state = gr.State(["MachineLearning-Lecture01.pdf"])
    upl_state = gr.State(False)
    user_state = gr.State(generate_unique_no)
    log_state = gr.State([])
    
    with gr.Tab("Chat Lounge"):
        chatbot = gr.Chatbot(label="Chat History", height=400)
        msg = gr.Textbox(label="User Input", placeholder="Enter your question")
        sendbtn = gr.Button(value="Ask AI", variant="primary")
        clear = gr.ClearButton([msg, chatbot], value="Clear History")
        
    with gr.Tab("AI Chronicles"):
        gr.Markdown("""Text from your uploaded files used to answer your question is shown below.
These pieces of text are extracted from a Vector DB.""")
        # gen_ques = gr.Markdown(format_current_query(""))
        # retr_docs = gr.Markdown(format_retrieved_docs(""))
        log_json_comp = gr.JSON([])
    
    sendbtn.click(ui_func, [msg, chatbot], [msg, chatbot], queue=False).then(
      ui_func_2, [chatbot, user_state, upl_state, log_state], 
      [msg, chatbot, log_state, log_json_comp]
    )
    
    msg.submit(ui_func, [msg, chatbot], [msg, chatbot], queue=False).then(
      ui_func_2, [chatbot, user_state, upl_state, log_state], 
      [msg, chatbot, log_state, log_json_comp]
    )
    
    with gr.Tab("Upload PDF"):
        gr.Markdown("""Upload your own PDF files of interest here. Your files will be temporarily saved
on the server for 15 mins. You can ask any question from your documents within that time. You can re 
upload any files within or after this time period. By default one of
[Andrew Ng's lecture transcript](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf)
is uploaded here.""")
        upload_button = gr.File(
            label="Upload Your Own Files For QnA", file_types=[".pdf"], file_count="multiple",
            elem_classes="upl-but"
        )
        current_files = gr.Markdown(format_file_names(docs_state.value))
        upload_button.upload(
            show_files_and_create_chain_ui, 
            [upload_button, user_state, docs_state, upl_state, current_files],
            [docs_state, upl_state, current_files]
        )

    with gr.Tab("Get Started"):
        gr.Markdown("""## Ready to get started?

1. Upload any PDF document of interest.
2. Then, ask any questions in the input field.
3. Simply press 'Enter' or 'Ask AI' Button to send your message. 
4. 'Chat Lounge': Entire Conversation History remains on this page!
5. 'AI chronicles': Interal System Logs.
6. 'Upload PDF': Upload your PDF files here. 😇

## Flow of System:  

1. Your question is sent to a Vector DB to fetch relevant documents.
2. Then these documents and your chat history is sent to the above LLM.
3. The LLM answers your question using this context.""")

demo.queue()
demo.launch(show_error=True)