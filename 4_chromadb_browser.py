#################################################################################################################################################################
###############################   CHROMADB BROWSER UI - View and Explore Your Local ChromaDB   ###############################################################
#################################################################################################################################################################

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import pandas as pd

load_dotenv()

st.set_page_config(
    page_title="ChromaDB Browser",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 ChromaDB Browser")
st.markdown("Browse and explore your local ChromaDB database")

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

@st.cache_resource
def get_embeddings():
    """Initialize embeddings model"""
    return OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
    )

@st.cache_resource
def get_vector_store():
    """Initialize ChromaDB vector store"""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=os.getenv("COLLECTION_NAME"),
        embedding_function=embeddings,
        persist_directory=os.getenv("DATABASE_LOCATION"),
    )

try:
    embeddings = get_embeddings()
    vector_store = get_vector_store()
    
    # Get collection info
    collection = vector_store._collection
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Choose a view:",
        ["📊 Overview", "📄 Browse Documents", "🔎 Search", "📈 Statistics"]
    )
    
    if page == "📊 Overview":
        st.header("Database Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                count = collection.count()
                st.metric("Total Documents", count)
            except:
                st.metric("Total Documents", "N/A")
        
        with col2:
            st.metric("Collection Name", os.getenv("COLLECTION_NAME", "N/A"))
        
        with col3:
            db_path = os.getenv("DATABASE_LOCATION", "N/A")
            st.metric("Database Path", db_path.split("\\")[-1] if "\\" in db_path else db_path)
        
        st.divider()
        
        # Collection metadata
        st.subheader("Collection Information")
        try:
            metadata = collection.metadata or {}
            if metadata:
                st.json(metadata)
            else:
                st.info("No metadata available for this collection")
        except Exception as e:
            st.warning(f"Could not retrieve metadata: {str(e)}")
    
    elif page == "📄 Browse Documents":
        st.header("Browse Documents")
        
        # Get all documents
        try:
            # Get a sample of documents
            results = vector_store.similarity_search("", k=100)  # Empty query to get documents
            
            if results:
                st.info(f"Showing {len(results)} documents. Use search for more specific queries.")
                
                # Pagination
                items_per_page = st.slider("Items per page", 5, 50, 10)
                total_pages = (len(results) - 1) // items_per_page + 1
                
                if total_pages > 1:
                    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                    start_idx = (page_num - 1) * items_per_page
                    end_idx = start_idx + items_per_page
                    page_results = results[start_idx:end_idx]
                else:
                    page_results = results
                
                # Display documents
                for idx, doc in enumerate(page_results, start=start_idx if total_pages > 1 else 0):
                    with st.expander(f"Document {idx + 1}", expanded=False):
                        st.markdown("**Content:**")
                        st.text_area(
                            "Document Content",
                            value=doc.page_content,
                            height=150,
                            key=f"content_{idx}",
                            label_visibility="collapsed"
                        )
                        
                        st.markdown("**Metadata:**")
                        st.json(doc.metadata)
                        
                        if hasattr(doc, 'id'):
                            st.caption(f"ID: {doc.id}")
            else:
                st.warning("No documents found in the database")
                
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            st.exception(e)
    
    elif page == "🔎 Search":
        st.header("Search Documents")
        
        # Search interface
        search_query = st.text_input("Enter your search query", placeholder="e.g., what is langchain")
        num_results = st.slider("Number of results", 1, 20, 5)
        
        if st.button("Search", type="primary") or search_query:
            if search_query:
                with st.spinner("Searching..."):
                    try:
                        results = vector_store.similarity_search(search_query, k=num_results)
                        
                        if results:
                            st.success(f"Found {len(results)} results")
                            
                            for idx, doc in enumerate(results, 1):
                                with st.expander(f"Result {idx}", expanded=idx == 1):
                                    st.markdown("**Content:**")
                                    st.text_area(
                                        "Document Content",
                                        value=doc.page_content,
                                        height=200,
                                        key=f"search_content_{idx}",
                                        label_visibility="collapsed"
                                    )
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Metadata:**")
                                        st.json(doc.metadata)
                                    with col2:
                                        if hasattr(doc, 'id'):
                                            st.markdown("**Document ID:**")
                                            st.code(doc.id)
                        else:
                            st.warning("No results found")
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                        st.exception(e)
            else:
                st.info("Enter a search query to begin")
    
    elif page == "📈 Statistics":
        st.header("Database Statistics")
        
        try:
            # Get all documents for analysis
            results = vector_store.similarity_search("", k=1000)  # Get up to 1000 for stats
            
            if results:
                # Content length statistics
                content_lengths = [len(doc.page_content) for doc in results]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Content Statistics")
                    df_stats = pd.DataFrame({
                        "Metric": ["Total Documents", "Average Length", "Min Length", "Max Length"],
                        "Value": [
                            len(results),
                            int(sum(content_lengths) / len(content_lengths)) if content_lengths else 0,
                            min(content_lengths) if content_lengths else 0,
                            max(content_lengths) if content_lengths else 0
                        ]
                    })
                    st.dataframe(df_stats, use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("Content Length Distribution")
                    st.bar_chart(pd.DataFrame({"Length": content_lengths}))
                
                # Metadata analysis
                st.subheader("Metadata Analysis")
                all_metadata_keys = set()
                for doc in results:
                    if doc.metadata:
                        all_metadata_keys.update(doc.metadata.keys())
                
                if all_metadata_keys:
                    metadata_df = []
                    for key in all_metadata_keys:
                        values = [doc.metadata.get(key, "N/A") for doc in results if doc.metadata]
                        unique_count = len(set(values))
                        metadata_df.append({
                            "Key": key,
                            "Unique Values": unique_count,
                            "Total Occurrences": len([v for v in values if v != "N/A"])
                        })
                    
                    st.dataframe(pd.DataFrame(metadata_df), use_container_width=True, hide_index=True)
                else:
                    st.info("No metadata found in documents")
            else:
                st.warning("No documents available for statistics")
                
        except Exception as e:
            st.error(f"Error generating statistics: {str(e)}")
            st.exception(e)
    
    # Footer
    st.divider()
    st.caption(f"Database Location: {os.getenv('DATABASE_LOCATION', 'N/A')} | Collection: {os.getenv('COLLECTION_NAME', 'N/A')}")

except Exception as e:
    st.error("Failed to connect to ChromaDB")
    st.exception(e)
    st.info("""
    **Troubleshooting:**
    1. Make sure your `.env` file has the correct settings:
       - `DATABASE_LOCATION` - path to your ChromaDB folder
       - `COLLECTION_NAME` - name of your collection
       - `EMBEDDING_MODEL` - your embedding model name
    2. Ensure the database exists and has been populated
    3. Check that Ollama is running if using Ollama embeddings
    """)

