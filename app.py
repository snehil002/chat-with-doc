import uuid
import gradio as gr
from helpers import process_input, create_db_by_loading_docs, initial_setup


initial_setup()



##### UI Functions ####

def format_file_names(file_paths):
    files_str = "\n".join(["* " + path for path in file_paths])
    return f"""**Currently Loaded Files:**
{files_str}"""

def format_current_query(query):
    return f"""**Current DB Query:**  
{query}"""

def format_retrieved_docs(docs):
    return f"""**Retrieved Documents:**  
{docs}"""

def generate_unique_no():
    uid = uuid.uuid4()
#     user_no_def+=1
    print("Generated User Id:", uid)
    return str(uid)

def format_info(user_state, docs_state, upl_state):
    return f"""**User ID:**  
{user_state}  
{format_file_names(docs_state)}  
**Has Uploaded:**  
{upl_state}"""

def ui_func(user, history):
    return "", history + [[user, None]]

def ui_func_2(history, user_no, has_uploaded):
#     print("User ID:", user_no)
    user = history.pop()[0]
    
    try:
        response, curr_query, retrieved_docs = process_input(
            user, history, user_no, has_uploaded
        )
        history += [[user, response]]
        return "", history, format_current_query(curr_query), format_retrieved_docs(retrieved_docs)
    
    except Exception as exp:
        print("Error!")
        print(exp.args)
        gr.Error(str(exp.args))
        return user, history, format_current_query(""), format_retrieved_docs("")

def show_files_and_create_chain_ui(files, user_state, docs_state, curr_files, upl_state):
    try:
        gr.Info("Your Files Are Uploaded!")

        file_paths = [file.name for file in files]
        upl_state = create_db_by_loading_docs(file_paths, user_state)
        file_paths = [file.split("/")[-1] for file in file_paths]
        formatted_paths = format_file_names(file_paths)

        gr.Info("Your Files Are Now Ready For QnA!")
        return file_paths, formatted_paths, upl_state
    
    except Exception as exp:
        print("Error!")
        print(exp.args)
        gr.Error(str(exp.args))
        return docs_state, curr_files, upl_state



####### UI Starts here #######

css="""
.upl-but{ height:50px; min-height: 150px }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("""# Chat with any Document of your interest
Ask any question in the input field. Press Enter to Send. 😇 History remains on this page!""")
    
    docs_state = gr.State(["MachineLearning-Lecture01.pdf"])
    upl_state = gr.State(False)
    user_state = gr.State(generate_unique_no)
    
    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot(label="Chat History", height=400)
        msg = gr.Textbox(label="User Input", placeholder="Enter your question")
        clear = gr.ClearButton([msg, chatbot])
        
    with gr.Tab("Vector DB"):
        gen_ques = gr.Markdown(format_current_query(""))
        retr_docs = gr.Markdown(format_retrieved_docs(""))
        
    msg.submit(ui_func, [msg, chatbot], [msg, chatbot], queue=False).then(
      ui_func_2, [chatbot, user_state, upl_state], [msg, chatbot, gen_ques, retr_docs]
    )
    
    with gr.Tab("Config"):
        upload_button = gr.File(
            label="Upload Your Own Files For QnA", file_types=[".pdf"], file_count="multiple",
            elem_classes="upl-but"
        )
        current_files = gr.Markdown(format_file_names(docs_state.value))
        upload_button.upload(
            show_files_and_create_chain_ui, 
            [upload_button, user_state, docs_state, current_files, upl_state],
            [docs_state, current_files, upl_state]
        )
    
    with gr.Tab("User info"):
        get_inf = gr.Button("Update User ID")
        user_id2 = gr.Markdown()
        get_inf.click(format_info, [user_state, docs_state, upl_state], user_id2)

demo.queue()
demo.launch(show_error=True)