"""Cached, lazily-loaded heavy resources (models, indexes, pipelines) for the Streamlit app."""

import logging
import streamlit as st

from local.app.config import SLM_MODEL_PATH

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Initialising privacy pipeline…")
def load_privacy_pipeline():
    from local.privacy.pipeline import PrivacyPipeline
    return PrivacyPipeline(spacy_model="en_core_web_lg")


@st.cache_resource(show_spinner="Loading Phi-3 GGUF…")
def load_inference_engine():
    try:
        from local.slm.inference import LocalModelInference
        return LocalModelInference(model_path=SLM_MODEL_PATH, ctx_size=4096, gpu_layers=0)
    except (FileNotFoundError, ImportError) as e:
        logger.info("Local inference unavailable: %s", e)
        return None


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


@st.cache_resource(show_spinner="Connecting to Pinecone…")
def load_pinecone_retriever():
    try:
        from local.rag.pinecone_index import PineconeRetriever
        return PineconeRetriever()
    except Exception as e:
        logger.warning("PineconeRetriever: %s", e)
        return None


@st.cache_resource(show_spinner="Loading cross-encoder…")
def load_reranker():
    try:
        from local.rag.reranker import CrossEncoderReranker
        return CrossEncoderReranker()
    except Exception as e:
        logger.warning("CrossEncoderReranker: %s", e)
        return None
