import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Create the prompt template
linkedin_prompt = PromptTemplate(
    input_variables=["rough_idea", "target_audience"],
    template="""
    Transform this rough idea into an engaging LinkedIn post:

    Rough idea: "{rough_idea}"
    Target audience: {target_audience}

    Make it:
    - Hook readers in the first line
    - Include 2-3 practical tips
    - End with an engaging question
    - Use emojis strategically
    - Keep it under 200 words

    Focus on providing real value!
    """
)

# Create the chain
# rough_idea → prompt → LLM → clean output
linkedin_chain = linkedin_prompt | llm | StrOutputParser()

#Use it!
result = linkedin_chain.invoke({
    "rough_idea": "I learned that most people don't know how to write good prompts for AI",
    "target_audience": "developers and tech professionals"
})

print("OPTIMIZED LINKEDIN POST:")
print(result)
