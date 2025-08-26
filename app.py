import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import ReActAgent
import asyncio
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

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
    
    
def sql_tool_func(db):
    """SQL query tool for database."""
    
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=db,
        tables=db.get_usable_table_names(),
        llm = Ollama(model="sqlcoder:7b", temperature=0)
    )

    sql_tool = QueryEngineTool.from_defaults(
        sql_query_engine,
        name="sql_tool",
        description=(
            "Useful for translating a natural language query into a SQL query over"
        ),
        
    )
    return sql_tool

async def run_agent(db, query):
    llm = Ollama(model="smollm2:135m")
    
    agent = ReActAgent(
        name = "Excel Analysis Agent",
        description = "Agent to analyze and query Excel or CSV data",
        system_prompt = "You are an expert data analyst. Use the provided tools to answer queries about the data.",
        tools =[sql_tool_func(db)],
        llm=llm,
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
                st.chat_message(msg["role"]).markdown(msg["content"])

            user_input = st.chat_input("Enter your question here...")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                try:
                    response = asyncio.run(run_agent(db, user_input))
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                    st.chat_message("assistant").markdown(str(response))
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.chat_message("assistant").markdown(f"Error: {str(e)}")

                     
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()