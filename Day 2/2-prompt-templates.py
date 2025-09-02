import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Instead of hardcoding prompts, create reusable templates
email_template = PromptTemplate(
    input_variables=["original_email", "tone", "sender_role"],
    template="""
    You're responding as a {sender_role}.

    Original email: "{original_email}"

    Generate a {tone} response that:
    1. Acknowledges their message
    2. Addresses their main point
    3. Provides a clear next step

    Keep it under 100 words and professional.
    """
)

# Now you can reuse this template for any email!
business_email = email_template.format(
    original_email="Hi! I loved your product demo. Can we schedule a call to discuss implementation?",
    tone="enthusiastic but professional",
    sender_role="startup founder"
)

casual_email = email_template.format(
    original_email="Hey, are you free for coffee this week?",
    tone="friendly and casual",
    sender_role="colleague"
)

print("BUSINESS RESPONSE:")
print(llm.invoke(business_email).content)

print("\nCASUAL RESPONSE:")
print(llm.invoke(casual_email).content)
