import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

promptTemplate = """
You're a business consultant. Analyze this scenario step-by-step and provide actionable recommendations.

Think through this systematically:

1. PROBLEM IDENTIFICATION
   → What are the core issues?
   → What's the root cause?

2. STAKEHOLDER ANALYSIS  
   → Who is affected?
   → What are their interests?

3. SOLUTION OPTIONS
   → What are 3 possible approaches?
   → Pros and cons of each?

4. RECOMMENDATION
   → Which solution is best and why?
   → Implementation steps?

Business Scenario: {business_scenario}

Let me work through this step-by-step:
"""

reasoning_prompt = PromptTemplate(
    input_variables=["business_scenario"],
    template=promptTemplate
)

# Create the chain
business_analyzer = reasoning_prompt | llm | StrOutputParser()

# Test with real scenario
scenario = """
Our e-commerce startup is losing customers after they add items to cart. 
Cart abandonment rate is 75%. Customer support says people complain about 
shipping costs being too high, but we can't reduce them without losing money.
"""

result = business_analyzer.invoke({"business_scenario": scenario})
print(result)
