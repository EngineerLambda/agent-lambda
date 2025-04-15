import os
import streamlit as st
from langchain.tools import tool
from langchain_google_genai import GoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from io import StringIO
from datetime import datetime


class QAParser(BaseModel):
    questions: List[str] = Field(..., description="List of questions generated")
    answers: List[str] = Field(..., description="List of answers corresponding to each question")

prompt_template_str = """
Please generate {number} questions and their corresponding answers based *only* on the following context provided. Do not use external knowledge.
Context:
{context}
Please provide the output in JSON format following this structure:
{format_instructions}
"""
prompt = PromptTemplate(
    template=prompt_template_str,
    input_variables=["number", "context"]
)
parser = PydanticOutputParser(pydantic_object=QAParser)
llm_qa = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ.get("GOOGLE_API_KEY"))
chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm_qa | parser


def create_csv(qa_data: QAParser) -> str:
    """Creates a CSV string from the QA data."""
    try:
        qa_df = pd.DataFrame({
            "Question": qa_data.questions,
            "Answer": qa_data.answers
        })
        csv_buffer = StringIO()
        qa_df.to_csv(csv_buffer, index=False, encoding='utf-8')
        return csv_buffer.getvalue()
    except Exception as e:
        raise


@tool("qa_generation", return_direct=False)
def qa_generation(number: int) -> str:
    """
    Generates a specified number of question-answer pairs based on document context,
    saves them as a CSV, adds an AI message to the session, and prepares for download.

    Args:
        number: The number of question-answer pairs to generate.

    Returns:
        A confirmation message string.
    """
    try:
        if 'file_docs' not in st.session_state or not st.session_state.file_docs:
            error_msg = "No document context found in session state. Please upload a document first."
            return error_msg

        qa_data = chain.invoke({"number": number, "context": st.session_state.file_docs})

        csv_content = create_csv(qa_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_pairs_{timestamp}.csv"
        st.session_state['downloadable_csv'] = {
            "filename": filename,
            "content": csv_content,
            "mime": "text/csv"
        }

        ai_message = f"Successfully generated {len(qa_data.questions)} question-answer pairs, download with button below now to avoid losing the data."
        return ai_message

    except Exception as e:
        error_response = f"An error occurred during question-answer generation: {e}"
        return error_response