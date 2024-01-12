import streamlit as st #all streamlit commands will be available through the "st" alias
import helper as glib #reference to local lib script


st.set_page_config(page_title="Changi AR Chatbot") #HTML title
st.title("Ask me anything about CAG's Annual Report") #page title

if 'vector_index' not in st.session_state: #see if the vector index hasn't been created yet
    with st.spinner("Indexing document..."): #show a spinner while the code in this with block runs
        st.session_state.vector_index = glib.get_index() #retrieve the index through the supporting library and store in the app's session cache

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

glib.initialise_convo()
for msg in glib.msgs.messages:
    st.chat_message(msg.type).write(msg.content)

input_text = st.text_area("Input text", label_visibility="collapsed") #display a multiline text box with no label
go_button = st.button("Go", type="primary") #display a primary button
print("st session state: ", st.session_state)


if go_button: #code in this if block will be run when the button is clicked
    # Append messages
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.markdown(input_text)

    # Get response
    with st.spinner("Working..."): #show a spinner while the code in this with block runs
        response_content = glib.get_rag_response(index=st.session_state.vector_index, question=input_text) #call the model through the supporting library
        
        st.write(response_content) #display the response content
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        

        