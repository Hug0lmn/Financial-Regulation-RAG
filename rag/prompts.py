"""
Prompt templates for the RAG system.

This module contains different prompt templates for various use cases
when working with IFRS standards documentation.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Default RAG prompt for IFRS standards
DEFAULT_RAG_TEMPLATE = """You are a financial expert assistant specializing in International Financial Reporting Standards (IFRS).
Your provide accurate answers derived directly using the provided context (from the IFRS documentation).

Context from IFRS standards:
{context}

Question: {question}

Instructions:
- Be precise and use technical terminology when appropriate
- Answer briefly in 3 sentences max
- Answer mainly using the information provided in the context above
- If the context doesn't contain enough information to answer the question, say "I cannot find sufficient information to answer this question accurately."
- Specify the parts where to find the answer
- If there are multiple aspects to the question, address each one

Answer:"""

# Detailed analysis prompt
DETAILED_ANALYSIS_TEMPLATE = """You are a senior financial reporting expert with deep knowledge of IFRS standards.

Analyze the following question using the provided context from IFRS documentation.

Context:
{context}

Question: {question}

Provide a detailed analysis that includes:
1. Direct answer to the question
2. Relevant IFRS standard references (IFRS 7, 9, or 13)
3. Key considerations or implications
4. Any related concepts that might be relevant

Analysis:"""

def get_prompt_template(template_type: str = "default") -> ChatPromptTemplate:
    """
    Get a prompt template based on the specified type.
    """
    templates = {
        "default": DEFAULT_RAG_TEMPLATE,
        "detailed": DETAILED_ANALYSIS_TEMPLATE
    }

    if template_type not in templates:
        raise ValueError(
            f"Unknown template type: {template_type}. "
            f"Available options: {', '.join(templates.keys())}"
        )

    return ChatPromptTemplate.from_template(templates[template_type])


def create_custom_prompt(template: str) -> ChatPromptTemplate:
    """
    Create a custom prompt template.
    """
    if "{context}" not in template or "{question}" not in template:
        raise ValueError(
            "Custom template must contain both {context} and {question} variables"
        )

    return ChatPromptTemplate.from_template(template)
