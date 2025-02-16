from os import getenv
from dotenv import load_dotenv
import gradio as gr
import glob

# Importaciones de LangChain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Cargar variables de entorno
load_dotenv()

# 1. Inicializamos el modelo de ChatOpenAI
llm = ChatOpenAI(
    openai_api_key=getenv("OPENROUTER_API_KEY"),
    openai_api_base=getenv("OPENROUTER_BASE_URL"),
    model_name="openai/gpt-4o",
    model_kwargs={
        "extra_headers": {
            "Helicone-Auth": f"Bearer " + getenv("HELICONE_API_KEY")
        }
    },
)

# 2. Cargar el documento PDF del Convenio del Metal desde la carpeta 'data'
pdf_files = glob.glob("data/*.pdf")
if not pdf_files:
    raise FileNotFoundError("No se encontraron archivos PDF en la carpeta 'data'.")

pdf_path = pdf_files[0]
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 3. Dividir el texto en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 4. Inicializar embeddings y base de datos vectorial
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(splits)


# 5. Definir herramientas para el agente

def analizar_situacion_laboral(query):
    """
    Evalúa si una situación laboral cumple con el convenio del metal.
    """
    relevant_docs = vector_store.similarity_search(query)
    if not relevant_docs:
        return "No encontré información relevante en los documentos."

    contexto = relevant_docs[0].page_content
    analisis_prompt = f"Analiza si esta situación cumple con el convenio del metal y responde en español:\n{contexto}"
    respuesta = llm.invoke(analisis_prompt)
    return respuesta.content


def calcular_compensacion(query):
    """
    Calcula indemnizaciones o compensaciones según el convenio del metal.
    """
    relevant_docs = vector_store.similarity_search(query)
    if not relevant_docs:
        return "No encontré información relevante en los documentos."

    contexto = relevant_docs[0].page_content
    compensacion_prompt = (f"Calcula la compensación económica para este caso según el convenio y responde en español"
                           f":\n{contexto}")
    respuesta = llm.invoke(compensacion_prompt)
    return respuesta.content


# 6. Crear herramientas para el agente
consulting_tools = [
    Tool(
        name="Analizar Situación Laboral",
        func=analizar_situacion_laboral,
        description="Evalúa si una situación laboral cumple con el convenio del metal."
    ),
    Tool(
        name="Calcular Compensación",
        func=calcular_compensacion,
        description="Calcula compensaciones o indemnizaciones según el convenio del metal."
    )
]

# 7. Inicializar el agente
agent_executor = initialize_agent(
    tools=consulting_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# 8. Integrar el agente con el chatbot
def chatbot(message, history):
    """
    Si el usuario usa "Agente: Analizar" o "Agente: Compensación", el agente realiza la evaluación.
    De lo contrario, usa la base de datos RAG para responder preguntas.
    """
    if message.lower().startswith("agente: analizar"):
        consulta = message.replace("agente: analizar", "").strip()
        respuesta = agent_executor.run(consulta)
    elif message.lower().startswith("agente: compensación"):
        consulta = message.replace("agente: compensación", "").strip()
        respuesta = agent_executor.run(consulta)
    else:
        relevant_docs = vector_store.similarity_search(message)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        final_prompt = (
            "Eres un asistente experto en el Convenio del Metal de Cádiz. "
            "Utiliza el siguiente contexto para responder de forma breve y concisa. "
            "Si no encuentras la información, responde que no la conoces.\n\n"
            f"Contexto:\n{context_text}\n\n"
            f"Pregunta: {message}\n"
            "Respuesta:"
        )
        respuesta = llm.invoke(final_prompt).content

    yield respuesta


# 9. Crear la interfaz de usuario con Gradio
demo = gr.ChatInterface(
    chatbot,
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Escribe tu mensaje aquí...", container=False, scale=7),
    title="ChatBot RAG - Convenio del Metal de Cádiz",
    description="Asistente virtual para consultar el Convenio del Metal de Cádiz y analizar situaciones laborales.",
    theme="ocean",
    examples=[
        "¿Cuáles son los derechos laborales en el convenio?",
        "Agente: Analizar una jornada de 12 horas de trabajo",
        "Agente: Compensación por despido injustificado"
    ],
    type="messages",
    editable=True,
    save_history=True,
)

if __name__ == "__main__":
    demo.queue().launch()
