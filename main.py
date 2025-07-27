import streamlit as st
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from tools import ImageCaptionTool, ObjectDetectionTool
from tempfile import NamedTemporaryFile
import os
import uuid


##### initialize agent #####
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", 
    k=5, 
    return_messages=True)

llm = ChatOpenAI(
    openai_api_key= "OPENAI_KEY ",
    model_name="gpt-3.5-turbo",
    temperature=0.0
)

agent = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent="chat-conversational-react-description", 
    verbose=True, 
    max_iterations=5,
    memory=conversational_memory,
    early_stopping_method="generate"
)



# set title 
st.title("Ask a question to an image")

#upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # display image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # ask question
    question = st.text_input("Ask a question about the image:")
    
    #### compute agent response ####

    # with NamedTemporaryFile(dir='.') as f:
    #     f.write(uploaded_file.getbuffer())
    #     image_path = f.name

    #     # write agent response
    #     if question and question != "":
    #         with st.spinner("Thinking..."):
    #             response = agent.run('{}, This is the image path: {}'.format(question, image_path))
    #             st.write(response)

    # Save uploaded image to 'images' folder
    os.makedirs("images", exist_ok=True)
    unique_filename = f"image_{uuid.uuid4().hex}.jpg"
    image_path = os.path.join("images", unique_filename)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if question:
        with st.spinner("Thinking..."):
            response = agent.run(f"{question}, This is the image path: {image_path}")
            st.write(response)
