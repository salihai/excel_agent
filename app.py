import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.agent.workflow import AgentWorkflow
import asyncio
from llama_index.core import Settings
import os
import dotenv
import openai
dotenv.load_dotenv()
openai.api_key = dotenv.get_key('.env', 'OPENAI_API_KEY')

def load_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        data = pd.read_csv(uploaded_file)
        
        engine = create_engine("sqlite:///:memory:")
        metadata_obj = MetaData()
        columns = [Column(col, String if data[col].dtype == "object" else Integer) for col in data.columns]
        table = Table("csv_table", metadata_obj, *columns)
        metadata_obj.create_all(engine)
        with engine.connect() as conn:
            for _, row in data.iterrows():
                conn.execute(table.insert().values(**row.to_dict()))
            conn.commit()
        
        sql_database = SQLDatabase(engine)
        
        return data, sql_database
    elif file_extension in ['xlsx', 'xls']:
        data = pd.read_excel(uploaded_file)
        
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        
        engine = create_engine("sqlite:///:memory:")
        metadata_obj = MetaData()
        for sheet_name, df in sheets.items():
            columns = [Column(col, String if df[col].dtype == "object" else Integer) for col in df.columns]
            table = Table(sheet_name, metadata_obj, *columns)
            metadata_obj.create_all(engine)
            with engine.connect() as conn:
                for _, row in df.iterrows():
                    conn.execute(table.insert().values(**row.to_dict()))
                conn.commit()
        sql_database = SQLDatabase(engine)
        
        return data, sql_database
    
    else:
        raise ValueError("Unsupported file format")
    
    
def sql_tool_func(db, query):
    """SQL query tool for database."""
    
    Settings.llm = HuggingFaceLLM(model_name="Ellbendls/Qwen-2.5-3b-Text_to_SQL",
        tokenizer_name="Ellbendls/Qwen-2.5-3b-Text_to_SQL",
        device_map="auto")
    
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=db,
        tables=db.get_usable_table_names(),
        llm = Settings.llm,
    )

    response = sql_query_engine.query(query)
    return response

async def run_agent(db, query):
    Settings.llm = Ollama(model="smollm2:135m", embed_model='local')
    
    agent = AgentWorkflow.from_tools_or_functions(
        [sql_tool_func(db, query)],
        llm=Settings.llm,
        system_prompt = "You are an expert data analyst. Use the provided tools to answer queries about the data.",
    )

    response = await agent.run(query)
    
    return response

def main():
    st.title("Excel Analysis Agent")
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        try:
            data, db = load_file(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(data.head())

            
            st.subheader("Ask a question about data")
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input("Enter your question here...")
            if user_input:
                st.chat_message("user").markdown(user_input)
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                try:
                    response = asyncio.run(run_agent(db, user_input))
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                    
                    with st.chat_message("assistant"):
                        st.markdown(str(response))
                        
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    with st.chat_message("assistant"):
                        st.markdown(error_message)

                     
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()