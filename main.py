import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import re


from prompts_chat_pdf import chat_prompt, CONDENSE_QUESTION_PROMPT


class PDFChatBot:

    def __init__(self):
        self.data_path = os.path.join('data')
        self.db_faiss_path = os.path.join('vectordb', 'db_faiss')
        #self.chat_prompt = PromptTemplate(template=chat_prompt, input_variables=['context', 'question'])
        #self.CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT
    
    def clean_and_minimize(self, text):
        # Define a regex pattern to remove non-alphanumeric characters and extra spaces
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra spaces
        return cleaned_text
    
    def create_vector_db(self):

        '''function to create vector db provided the pdf files'''

        loader = DirectoryLoader(self.data_path,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

        documents = loader.load()

        for i in documents:
            i.page_content = self.clean_and_minimize(i.page_content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,
                                                   chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(self.db_faiss_path)

    def load_llm(self):
        # Load the locally downloaded model here
        llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_file="llama-2-7b-chat.ggmlv3.q4_1.bin",
            max_new_tokens=4000,
            temperature=0.7,
        )
        return llm

    def conversational_chain(self):

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(self.db_faiss_path, embeddings)
        # initializing the conversational chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversational_chain = ConversationalRetrievalChain.from_llm( llm=self.load_llm(),
                                                                      retriever=db.as_retriever(search_kwargs={"k": 3}),
                                                                      verbose=True,
                                                                      memory=memory
                                                                      )

        return conversational_chain

def intialize_chain():
    bot = PDFChatBot()
    bot.create_vector_db()
    conversational_chain = bot.conversational_chain()
    return conversational_chain

chat_history = []

chain = intialize_chain()

while(True):
    query = input('User: ')
    response = chain({"question": query, "chat_history": chat_history})
    chat_history.append(response["answer"])  # Append the answer to chat history
    print(response["answer"])