from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core.query_pipeline import (QueryPipeline as QP, Link, InputComponent)
import gradio as gr
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os

# API KEY do GROQ
api_key = os.getenv("secret_key")

# Configuração do modelo 
llm = Groq(model="llama3-70b-8192", api_key=api_key)

# Pipeline de consulta
'''Function to generate the description of the dataframe columns'''
def get_column_descriptions(df):
    description = '\n'.join([f"`{col}`: {str(df[col].dtype)}" for col in df.columns])
    return "Here are the details of the dataframe columns:\n" + description

'''Definition of pipeline modules'''
def query_pipeline(df):
    instruction_str = (
        "1. Convert the query to executable Python code using Pandas.\n"
        "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
        "3. The code should represent a solution to the query.\n"
        "4. PRINT ONLY THE EXPRESSION.\n"
        "5. Do not enclose the expression in quotes.\n")

    pandas_prompt_str = (
        "You are working with a pandas dataframe in Python called `df`.\n"
        "{column_details}\n\n"
        "This is the result of `print(df.head())`:\n"
        "{df_str}\n\n"
        "Follow these instructions:\n"
        "{instruction_str}\n"
        "Query: {query_str}\n\n"
        "Expression:"
)

    response_synthesis_prompt_str = (
       "Given an input question, act as a data analyst and formulate a response from the query results.\n"
       "Answer in a natural way, without introductions like 'The answer is:' or something similar.\n"
       "Query: {query_str}\n\n"
       "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
       "Pandas Output: {pandas_output}\n\n"
       "Answer: \n\n"
       "At the end, display the code used to generate the answer, in the format: The code used was `{pandas_instructions}`"
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str,
    df_str=df.head(5),
    column_details=get_column_descriptions(df)
)

    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    '''Query Pipeline Creation'''
    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": llm,
        },
        verbose=True,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
            Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output"),
        ]
    )
    qp.add_link("response_synthesis_prompt", "llm2")
    return qp

# Função para carregar os dados
def load_data(file_path, df_state):
    if file_path is None or file_path == "":
        return "Please upload a CSV file to analyze.", pd.DataFrame(), df_state
    try:
        df = pd.read_csv(file_path)
        return "File loaded successfully!", df.head(), df
    except Exception as e:
        return f"Error loading file: {str(e)}", pd.DataFrame(), df_state

# Função para processar a pergunta
def process_question(question, df_state):
    if df_state is not None and question:
        qp = query_pipeline(df_state)
        response = qp.run(query_str=question)
        return response.message.content
    return ""

# Função para adicionar a pergunta e a resposta ao histórico
def add_to_history(question, response, history_state):
    if question and response:
        history_state.append((question, response))
        gr.Info("Added to PDF!", duration=2)
        return history_state

# Função para gerar o PDF
def generate_pdf(history_state):
    if not history_state:
        return "No data to add to the PDF.", None

    # Gerar nome de arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    pdf_path = f"questions_answers_report_{timestamp}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    for question, response in history_state:
        pdf.set_font("Arial", "B", 14)
        pdf.multi_cell(0, 8, txt=question)
        pdf.ln(2)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, txt=response)
        pdf.ln(6)

    pdf.output(pdf_path)
    return pdf_path

# Função para limpar a pergunta e a resposta
def clear_question_response():
    return "", ""

# Função para resetar a aplicação
def reset_application():
    return None, "The application has been reset. Please upload a new CSV file.", pd.DataFrame(), "", None, [], ""

# Criação da interface gradio
with gr.Blocks(theme="Soft") as app:

    # Título da app
    gr.Markdown("# Analyzing data🔎🎲")

    # Descrição
    gr.Markdown("""
    Upload a CSV file and ask questions about the data. For each question, you can
    view the answer and, if you wish, add this interaction to the final PDF, just click
    on "Add to PDF history". To ask a new question, click on "Clear question and result".
    After defining the questions and answers in the history, click on "Generate PDF". This will allow you to
    download a PDF with the complete record of your interactions. If you want to analyze a new dataset,
    just click on "I want to analyze another dataset" at the bottom of the page.
    """)

    # Campo de entrada de arquivos
    file_input = gr.File(file_count="single", type="filepath", label="Upload CSV")

    # Status de upload
    upload_status = gr.Textbox(label="Upload Status:")

    # Tabela de dados
    data_table = gr.DataFrame()

    # Exemplos de perguntas
    gr.Markdown("""
    Example questions:
    1. What is the number of records in the file?
    2. What are the data types of the columns?
    3. What are the descriptive statistics of the numerical columns?
    """)

    # Campo de entrada de texto
    question_input = gr.Textbox(label="Enter your question about the data")

    # Botão de envio posicionado após a pergunta
    submit_button = gr.Button("Submit")

    # Componente de resposta
    response_output = gr.Textbox(label="Answer")

    # Botões para limpar a pergunta e a resposta, adicionar ao historico e gerar o PDF
    with gr.Row():
        clear_button = gr.Button("Clear question and result")
        add_to_pdf_button = gr.Button("Add to PDF history")
        generate_pdf_button = gr.Button("Generate PDF")

    # Componente de download
    pdf_file = gr.File(label="Download PDF")

    # Botão para resetar a aplicação
    reset_button = gr.Button("I want to analyze another dataset!")

    # Gerenciamento de estados
    df_state = gr.State(value=None)
    history_state = gr.State(value=[])

    # Conectando funções aos componentes
    file_input.change(fn=load_data, inputs=[file_input, df_state], outputs=[upload_status, data_table, df_state])
    submit_button.click(fn=process_question, inputs=[question_input, df_state], outputs=response_output)
    clear_button.click(fn=clear_question_response, inputs=[], outputs=[question_input, response_output])
    add_to_pdf_button.click(fn=add_to_history, inputs=[question_input, response_output, history_state], outputs=history_state)
    generate_pdf_button.click(fn=generate__pdf, inputs=[history_state], outputs=pdf_file)
    reset_button.click(fn=reset_application, inputs=[], outputs=[file_input, upload_status, data_table, response_output, pdf_file, history_state, question_input])

if __name__ == "__main__":
    app.launch()
