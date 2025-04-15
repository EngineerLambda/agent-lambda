# Lambda Agent: AI Assistant

Lambda Agent is an AI assistant that leverages Langchain and Google's Gemini models to provide intelligent responses based on user queries. It supports chat history, document uploads for context, and various tools for enhanced functionality.

## Features

*   **Chat Interface:** Interactive chat interface for users to ask questions and receive responses.
*   **Document Upload:** Users can upload PDF or DOCX files to provide context for the AI assistant.
*   **Tool Integration:** Integrates with tools like Tavily Search for general knowledge and document search for content-specific answers.
*   **Chat History:** Maintains chat history for ongoing conversations.
*   **User Authentication:** Utilizes Google OAuth for user authentication.
*   **Feedback Mechanism:** Allows users to provide feedback on the assistant's performance.
*   **Q\&A Generation:** Generates question and answer pairs based on uploaded documents.

## Code Structure

*   `app.py`: Main Streamlit application file, handles UI, user authentication, and chat logic.
*   `agent.py`: Defines the AI agent, prompt templates, and tool configurations.
*   `ext_tools/`: Contains custom tools like `qa_tool.py` for question-answer generation and `instant_rag.py` for document retrieval.
*   `utils/`: Includes utility functions for database interactions (`database.py`), account management (`account.py`), and feedback handling (`feedback.py`).

## Dependencies

*   `langchain`: Framework for building language model applications.
*   `langchain_google_genai`: Google Gemini integration for Langchain.
*   `pymongo`: MongoDB driver for storing chat history and feedback.
*   `streamlit`: Python library for creating interactive web applications.
*   `tavily-python`: Tavily Search API for web searches.
*   `PyPDF2`, `python-docx`: Libraries for processing PDF and DOCX files.
*   `Authlib`: Authentication library.
*   `dropbox`: Dropbox integration.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Secrets:**

    *   Create a `.streamlit/secrets.toml` file.
    *   Add the following secrets, replacing the placeholder values with your actual credentials:

        ```toml
        "GROQ_API_KEY" = "YOUR_GROQ_API_KEY"
        "DATABASE_URL" = "YOUR_MONGODB_CONNECTION_STRING"
        "TAVILY_API_KEY" = "YOUR_TAVILY_API_KEY"
        "ADMIN_EMAIL" = "YOUR_ADMIN_EMAIL"
        "GOOGLE_API_KEY" = "YOUR_GOOGLE_API_KEY"
        "DROPBOX_ACCESS_TOKEN" = "YOUR_DROPBOX_ACCESS_TOKEN"

        [auth]
        "redirect_uri" = "http://localhost:8501/oauth2callback"
        "cookie_secret" = "YOUR_COOKIE_SECRET"
        "client_id" = "YOUR_GOOGLE_CLIENT_ID"
        "client_secret" = "YOUR_GOOGLE_CLIENT_SECRET"
        "server_metadata_<!-- filepath: /home/engineerlambda/lambda_agent/README.md -->
# Lambda Agent: AI Assistant

Lambda Agent is an AI assistant that leverages Langchain and Google's Gemini models to provide intelligent responses based on user queries. It supports chat history, document uploads for context, and various tools for enhanced functionality.

## Features

*   **Chat Interface:** Interactive chat interface for users to ask questions and receive responses.
*   **Document Upload:** Users can upload PDF or DOCX files to provide context for the AI assistant.
*   **Tool Integration:** Integrates with tools like Tavily Search for general knowledge and document search for content-specific answers.
*   **Chat History:** Maintains chat history for ongoing conversations.
*   **User Authentication:** Utilizes Google OAuth for user authentication.
*   **Feedback Mechanism:** Allows users to provide feedback on the assistant's performance.
*   **Q\&A Generation:** Generates question and answer pairs based on uploaded documents.

## Code Structure

*   `app.py`: Main Streamlit application file, handles UI, user authentication, and chat logic.
*   `agent.py`: Defines the AI agent, prompt templates, and tool configurations.
*   `ext_tools/`: Contains custom tools like `qa_tool.py` for question-answer generation and `instant_rag.py` for document retrieval.
*   `utils/`: Includes utility functions for database interactions (`database.py`), account management (`account.py`), and feedback handling (`feedback.py`).

## Dependencies

*   `langchain`: Framework for building language model applications.
*   `langchain_google_genai`: Google Gemini integration for Langchain.
*   `pymongo`: MongoDB driver for storing chat history and feedback.
*   `streamlit`: Python library for creating interactive web applications.
*   `tavily-python`: Tavily Search API for web searches.
*   `PyPDF2`, `python-docx`: Libraries for processing PDF and DOCX files.
*   `Authlib`: Authentication library.
*   `dropbox`: Dropbox integration.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Secrets:**

    *   Create a `.streamlit/secrets.toml` file.
    *   Add the following secrets, replacing the placeholder values with your actual credentials:

        ```toml
        "DATABASE_URL" = "YOUR_MONGODB_CONNECTION_STRING"
        "TAVILY_API_KEY" = "YOUR_TAVILY_API_KEY"
        "ADMIN_EMAIL" = "YOUR_ADMIN_EMAIL"
        "GOOGLE_API_KEY" = "YOUR_GOOGLE_API_KEY"

        [auth]
        "redirect_uri" = "http://localhost:8501/oauth2callback"
        "cookie_secret" = "YOUR_COOKIE_SECRET"
        "client_id" = "YOUR_GOOGLE_CLIENT_ID"
        "client_secret" = "YOUR_GOOGLE_CLIENT_SECRET"
        "server_metadata_url" = "https://accounts.google.com/.well-known/openid-configuration"
        ```
3.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
4.  **Access the Application:**
    Open your web browser and navigate to `http://localhost:8501`.
5.  **Authenticate:**
    Click on the "Login with Google" button to authenticate using your Google account. Ensure that the email used is the same as the one specified in `ADMIN_EMAIL` in the secrets file.
6.  **Upload Documents:**
    Use the document upload feature to provide context for the AI assistant. Supported formats are PDF and DOCX.
7.  **Ask Questions:**
    Type your questions in the chat interface and receive intelligent responses from the AI assistant.
8.  **Provide Feedback:**
    Use the feedback mechanism to rate the assistant's performance and provide comments.
9.  **Generate Q\&A:**
    Use the Q\&A generation feature to create question and answer pairs based on the uploaded documents.
10. **Search the Web:**
    Use the Tavily Search tool to search the web for general knowledge queries.
11. **View Chat History:**
    Access the chat history to review past interactions with the AI assistant.
12. **Logout:**
    Click on the "Logout" button to end your session.
## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

