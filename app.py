import streamlit as st
import upload
import localrag

st.set_page_config(
    page_title="AskPDF - Chat with Your Documents",
    layout="wide",
    page_icon="📄",
)

# col1, col2 = st.columns([10, 1])
# with col2:
#     st.image("ask_pdf.png", width=90)

st.title("📄 AskPDF: Intelligent Document Chat")


st.markdown(
    "Welcome to **AskPDF**, your AI-powered assistant for interacting with PDF, TXT, or JSON files. "
    "Upload a document and ask questions – the assistant will retrieve the most relevant context for your query."
)

# Initialize conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = []

import base64


# Convert image to base64
def local_image_to_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


with st.sidebar:
    img_base64 = local_image_to_base64("ask_pdf.png")
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{img_base64}" style="height:180px; border-radius:10px;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.header("📤 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF, TXT, or JSON file", type=["pdf", "txt", "json"]
    )
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            chunks = upload.process_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            chunks = upload.process_txt(uploaded_file)
        elif uploaded_file.type == "application/json":
            chunks = upload.process_json(uploaded_file)
        else:
            st.sidebar.error("❌ Unsupported file type")
            chunks = []

        if chunks:
            upload.append_to_vault(chunks)
            st.sidebar.success(f"✅ {len(chunks)} chunks added to vault.")


# Load document and embeddings
@st.cache_data(show_spinner=False)
def load_data():
    content = localrag.load_vault()
    embeddings = localrag.get_vault_embeddings(content)
    return content, embeddings


vault_content, vault_embeddings = load_data()

system_message = (
    "You are a helpful assistant that extracts the most useful information "
    "from a given text and provides extra relevant info."
)

# Chat Form
st.markdown("## 💬 Ask a Question About Your Document")
with st.form("chat_form", clear_on_submit=False):
    user_input = st.text_input("Type your question here:")
    col1, col2 = st.columns([1, 1])
    send = col1.form_submit_button("📨 Send")
    clear = col2.form_submit_button("🧹 Clear Chat")

    if clear:
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

    if send and user_input.strip():
        with st.spinner("🧠 Thinking..."):
            response = localrag.ollama_chat(
                user_input,
                system_message,
                vault_embeddings,
                vault_content,
                st.session_state.conversation,
            )
        st.session_state.conversation.append({"role": "user", "content": user_input})
        st.session_state.conversation.append({"role": "assistant", "content": response})

# Chat History
if st.session_state.conversation:
    st.markdown("---")
    st.subheader("📝 Conversation History")
    for msg in st.session_state.conversation:
        role = "🧑‍💼 You" if msg["role"] == "user" else "🤖 Assistant"
        st.markdown(f"**{role}:** {msg['content']}")
