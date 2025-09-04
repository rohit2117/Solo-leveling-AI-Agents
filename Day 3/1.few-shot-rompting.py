import os

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

promptTemplate = """
Analyze the sentiment of product reviews. Classify as POSITIVE, NEGATIVE, 
or NEUTRAL with a confidence score.

Examples:

Review: "This app is amazing! It saved me hours of work and the interface is intuitive."
Sentiment: POSITIVE
Confidence: 95%
Reason: Strong positive language ("amazing", "saved me hours")

Review: "The app works okay but crashes sometimes. Could be better."
Sentiment: NEUTRAL  
Confidence: 80%
Reason: Mixed feedback - works but has issues

Review: "Terrible experience. Constant bugs, poor customer service, waste of money."
Sentiment: NEGATIVE
Confidence: 98%
Reason: Multiple strong negative indicators

Now analyze this review:
Review: "{review}"
Sentiment: 
Confidence: 
Reason:
"""

# Few-shot prompt template
sentiment_analyzer = PromptTemplate(
    input_variables=["review"],
    template=promptTemplate
)

# Test it
test_review = "The features are decent but the pricing is way too high for what you get"

result = llm.invoke(sentiment_analyzer.format(review=test_review))
print(result.content)
