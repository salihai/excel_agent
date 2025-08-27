# Excel Analysis Agent

Excel Analysis Agent is a Streamlit application that lets you upload Excel or CSV files, loads your data into a temporary SQLite database, and enables you to ask natural language questions about your data using a local LLM agent.

## Features

- Upload `.csv`, `.xlsx`, or `.xls` files
- Preview your data in the browser
- Chat interface for asking questions about your data
- Uses local LLMs via Ollama (no OpenAI key required)
- SQL queries are generated and executed automatically

## Requirements

- Python 3.10+
- Streamlit
- pandas
- SQLAlchemy
- llama-index

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/excel_agent.git
    cd excel_agent
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) Set up environment variables:**
    - If you need to configure API keys or other secrets, create a `.env` file in the project root.

## Usage


### Run Locally

1. **Start Ollama manually (if installed):**
    ```bash
    ollama serve
    ollama pull smollm2:135m
    ollama pull sqlcoder:7b
    ```

2. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Example

- Upload a file like `sales.xlsx`.
- Ask:  
  ```
  What is the total sales for 2024?
  ```
- The agent will analyze your data and return the answer in the chat interface.

## Notes

- The app uses local models via Ollama. No OpenAI API key is required.
- All data is loaded into a temporary SQLite database for querying.
- For large files or models, performance may vary depending on your hardware.
- Chat history is visible and persistent during the session.

## License

MIT License

---