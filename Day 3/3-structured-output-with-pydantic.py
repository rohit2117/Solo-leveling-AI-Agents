import os

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import List

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Define the structure you want
class CompetitorAnalysis(BaseModel):
    company_name: str = Field(description="Name of the competitor")
    strengths: List[str] = Field(description="Top 3 competitive advantages")
    weaknesses: List[str] = Field(description="Top 3 areas they struggle with")
    market_share: str = Field(description="Estimated market share percentage")
    threat_level: str = Field(description="HIGH, MEDIUM, or LOW threat level")
    key_insight: str = Field(description="Most important strategic insight")

# Create parser
parser = PydanticOutputParser(pydantic_object=CompetitorAnalysis)

promptTemplate = """
Analyze this competitor based on publicly available information.
Be specific and actionable in your analysis.

Company: {company_name}
Industry Context: {industry_context}

{format_instructions}
"""
# Build prompt with format instructions
competitor_prompt = PromptTemplate(
    template=promptTemplate,
    input_variables=["company_name", "industry_context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create the analysis chain
competitor_analyzer = competitor_prompt | llm | parser

# Test it
analysis = competitor_analyzer.invoke({
    "company_name": "Notion",
    "industry_context": "Productivity and collaboration tools for knowledge workers"
})

print(f"Company: {analysis.company_name}")
print(f"Threat Level: {analysis.threat_level}")
print(f"Strengths: {', '.join(analysis.strengths)}")
print(f"Key Insight: {analysis.key_insight}")
