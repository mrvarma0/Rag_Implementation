import unittest
from unittest.mock import patch, MagicMock
from fast_api_app import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain, handle_userinput
from PyPDF2 import PdfReader
import streamlit as st

class TestPDFProcessing(unittest.TestCase):

    @patch('PyPDF2.PdfReader')
    def test_get_pdf_text(self, MockPdfReader):
        # Mock PDF Reader
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = "This is the first page."
        mock_pdf.pages[1].extract_text.return_value = "This is the second page."
        MockPdfReader.return_value = mock_pdf
        
        pdf_docs = ["dummy.pdf"]
        text = get_pdf_text(pdf_docs)
        self.assertEqual(text, "This is the first page.This is the second page.")
    
    def test_get_text_chunks(self):
        text = "This is a long text that needs to be split into chunks."
        chunks = get_text_chunks(text)
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    @patch('langchain_community.vectorstores.FAISS')
    def test_get_vectorstore(self, MockFAISS, MockHuggingFaceEmbeddings):
        # Mock embeddings and FAISS
        mock_embeddings = MagicMock()
        MockHuggingFaceEmbeddings.return_value = mock_embeddings
        mock_faiss = MagicMock()
        MockFAISS.from_texts.return_value = mock_faiss
        
        text_chunks = ["chunk1", "chunk2"]
        vectorstore = get_vectorstore(text_chunks)
        self.assertEqual(vectorstore, mock_faiss)

    @patch('transformers.pipeline')
    @patch('langchain_community.llms.huggingface_pipeline.HuggingFacePipeline')
    @patch('langchain.memory.ConversationBufferMemory')
    @patch('langchain.chains.ConversationalRetrievalChain')
    def test_get_conversation_chain(self, MockConversationalRetrievalChain, MockConversationBufferMemory, MockHuggingFacePipeline, MockPipeline):
        # Mock pipeline, HuggingFacePipeline, and ConversationalRetrievalChain
        mock_pipe = MagicMock()
        MockPipeline.return_value = mock_pipe
        mock_llm = MagicMock()
        MockHuggingFacePipeline.return_value = mock_llm
        mock_memory = MagicMock()
        MockConversationBufferMemory.return_value = mock_memory
        mock_chain = MagicMock()
        MockConversationalRetrievalChain.from_llm.return_value = mock_chain

        vectorstore = MagicMock()
        chain = get_conversation_chain(vectorstore)
        self.assertEqual(chain, mock_chain)

    @patch('streamlit.write')
    @patch('app.st.session_state')
    def test_handle_userinput(self, MockSessionState, MockStWrite):
        # Mock session state and st.write
        mock_response = {
            'chat_history': [
                MagicMock(content='User question'),
                MagicMock(content='Bot answer')
            ]
        }
        MockSessionState.conversation.return_value = mock_response
        
        handle_userinput("What is the topic of this PDF?")
        
        # Assert that st.write was called twice, once for user and once for bot
        self.assertEqual(MockStWrite.call_count, 2)
        MockStWrite.assert_any_call('<div class="user"><b>User:</b> What is the topic of this PDF?</div>', unsafe_allow_html=True)
        MockStWrite.assert_any_call('<div class="bot"><b>Bot:</b> Bot answer</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    unittest.main()