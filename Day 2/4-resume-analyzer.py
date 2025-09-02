import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class ResumeAnalyzerAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )

        # The analysis prompt template
        self.analysis_prompt = PromptTemplate(
            input_variables=["resume_text", "job_role"],
            template="""
            Analyze this resume for a {job_role} position:

            Resume:
            {resume_text}

            Provide specific feedback on:

            STRENGTHS (2-3 points):
            → What stands out positively

            AREAS TO IMPROVE (2-3 points):
            → Specific actionable suggestions

            MISSING ELEMENTS:
            → What's typically expected but missing

            OVERALL SCORE: X/10

            Keep feedback constructive and specific!
            """
        )

        # Create the analysis chain
        self.analyzer = self.analysis_prompt | self.llm | StrOutputParser()

    def analyze(self, resume_text, job_role):
        return self.analyzer.invoke({
            "resume_text": resume_text,
            "job_role": job_role
        })


# Test it with a dummy resume content
analyzer = ResumeAnalyzerAgent()

sample_resume = """
Rohit Sharma
Software Developer

Experience:
- Built 3 web applications using React and Node.js
- Worked with databases like MySQL
- 2 years experience at tech startup

Skills: JavaScript, Python, React, Node.js

Education: Computer Science degree
"""

result = analyzer.analyze(sample_resume, "Senior Frontend Developer")
print(result)
