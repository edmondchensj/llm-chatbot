import os
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms.bedrock import Bedrock
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv

load_dotenv() 
# os.environ["BWB_PROFILE_NAME"]
# os.environ["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"]

msgs = StreamlitChatMessageHistory(key="special_app_key")
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)


def initialise_convo():
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

def get_llm():
    model_kwargs = { #AI21
        "maxTokens": 1024, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": [], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 } 
    }
    
    # llm = Bedrock(
    #     credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
    #     region_name=os.environ.get("us-east"), #sets the region name (if not the default)
    #     endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
    #     model_id="ai21.j2-ultra-v1", #set the foundation model
    #     model_kwargs=model_kwargs) #configure the properties for Claude
    llm = OpenAI()
    return llm

def get_pdf_files():
    ''' Fetch all pdf files from pdfs folder and return list of filepaths '''
    #return [f for f in os.listdir("pdfs") if os.path.isfile(f)]
    return ["pdfs/annualreport-fy2223.pdf"] #["pdfs/cag-annual-leave.pdf"]


def get_index(): #creates and returns an in-memory vector store to be used in the application
    
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
    #     region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
    #     endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
    # ) #create a Titan Embeddings client
    embeddings = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter( #create a text splitter
        separators=["\n\n", "\n", ".", " "], #split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
        chunk_size=1000, #divide into 1000-character chunks using the separators above
        chunk_overlap=100 #number of characters that can overlap with previous chunk
    )
    
    index_creator = VectorstoreIndexCreator( #create a vector store factory
        vectorstore_cls=FAISS, #use an in-memory vector store for demo purposes
        embedding=embeddings, #use Titan embeddings
        text_splitter=text_splitter, #use the recursive text splitter
    )
    

    pdf_paths = get_pdf_files()
    
    loaders = [PyPDFLoader(path) for path in pdf_paths]
    print('pdf paths: ', pdf_paths)
    print('loaders: ', loaders)
    index_from_loader = index_creator.from_loaders(loaders) #create an vector store index from the loaded PDF
    
    return index_from_loader #return the index to be cached by the client app


def get_rag_response(index, question): #rag client function
    msgs.add_user_message(question)

    llm = get_llm()
    
    response_text = index.query(question=question, llm=llm) #search against the in-memory index, stuff results into a prompt and send to the llm
    msgs.add_ai_message(response_text)
    return response_text

if __name__ == "__main__":
    get_index()