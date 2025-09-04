import os
from typing import List

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Define structured outputs
class JobAnalysis(BaseModel):
    role_title: str = Field(description="Job title")
    key_requirements: List[str] = Field(description="Top 5 must-have skills/requirements")
    company_values: List[str] = Field(description="Company culture/values mentioned")
    pain_points: List[str] = Field(description="Problems this role needs to solve")
    experience_level: str = Field(description="ENTRY, MID, SENIOR, or EXECUTIVE")
    application_strategy: str = Field(description="Best approach to stand out for this role")


class CoverLetterContent(BaseModel):
    opening_hook: str = Field(description="Compelling first paragraph")
    body_paragraphs: List[str] = Field(description="2-3 body paragraphs highlighting relevant experience")
    closing_call_to_action: str = Field(description="Strong closing paragraph")
    tone_style: str = Field(description="Professional tone to match company culture")


class JobApplicationAssistant:
    def __init__(self):
        self.llm = llm

        # Job analysis chain
        self.job_parser = PydanticOutputParser(pydantic_object=JobAnalysis)

        self.job_analysis_prompt = PromptTemplate(
            template="""
You're a senior recruiter with 15+ years experience. Analyze this job posting strategically.

Job Posting:
{job_posting}

Extract key insights that will help a candidate craft a winning application:

{format_instructions}
""",
            input_variables=["job_posting"],
            partial_variables={"format_instructions": self.job_parser.get_format_instructions()}
        )

        # Cover letter generation chain
        self.cover_letter_parser = PydanticOutputParser(pydantic_object=CoverLetterContent)

        self.cover_letter_prompt = PromptTemplate(
            template="""
You're a career coach who's helped hundreds get dream jobs. 

Create a compelling cover letter structure based on this analysis:

Job Analysis: {job_analysis}
Candidate Background: {candidate_background}

Rules for an outstanding cover letter:
1. HOOK: Start with something specific about the company/role that excites you
2. CONNECT: Show how your experience directly solves their pain points  
3. PROVE: Use specific achievements with numbers when possible
4. PERSONALITY: Match their company culture tone
5. ACTION: End with confident next steps

{format_instructions}
""",
            input_variables=["job_analysis", "candidate_background"],
            partial_variables={"format_instructions": self.cover_letter_parser.get_format_instructions()}
        )

        # Interview prep chain
        self.interview_prep_prompt = PromptTemplate(
            template="""
You're an interview coach. Based on this job analysis, prepare the candidate.

Job Analysis: {job_analysis}

Create comprehensive interview preparation:

LIKELY INTERVIEW QUESTIONS (Top 5):
List the most probable questions for this specific role and company.

STAR METHOD ANSWERS:
For each question, provide a framework using:
- Situation: Context you'll reference
- Task: What you needed to accomplish  
- Action: Specific steps you took
- Result: Measurable outcome

QUESTIONS TO ASK THEM:
5 intelligent questions that show you understand their business challenges.

RED FLAGS TO AVOID:
Common mistakes candidates make for this type of role.
""",
            input_variables=["job_analysis"]
        )

    def analyze_job(self, job_posting: str):
        """Analyze job posting for key insights"""
        analysis_chain = self.job_analysis_prompt | self.llm | self.job_parser
        return analysis_chain.invoke({"job_posting": job_posting})

    def generate_cover_letter(self, job_analysis: JobAnalysis, candidates_background: str):
        """Generate customized cover letter structure"""
        cover_letter_chain = self.cover_letter_prompt | self.llm | self.cover_letter_parser
        return cover_letter_chain.invoke({
            "job_analysis": job_analysis.dict(),
            "candidate_background": candidates_background
        })

    def prepare_interview(self, job_analysis: JobAnalysis):
        """Generate interview preparation materials"""
        interview_chain = self.interview_prep_prompt | self.llm | StrOutputParser()
        return interview_chain.invoke({"job_analysis": job_analysis.dict()})


# Test with real job posting
assistant = JobApplicationAssistant()

sample_job_posting = """
Senior Frontend Developer - TechFlow Solutions

We're a fast-growing fintech startup revolutionizing small business lending. 
Looking for a Senior Frontend Developer to lead our user experience transformation.

Requirements:
- 5+ years React.js experience
- TypeScript proficiency  
- Experience with financial applications
- Strong UX/UI collaboration skills
- Startup mentality and ownership mindset

You'll be:
- Building intuitive lending dashboards
- Optimizing application flow (currently 45% drop-off rate)
- Working directly with CEO and design team
- Mentoring junior developers

We offer equity, unlimited PTO, and the chance to impact thousands of small businesses.
"""

candidate_background = """
I'm a frontend developer with 6 years experience. Built 3 React applications for e-commerce companies. 
Expert in TypeScript and have worked on checkout flows optimization. Led a team of 2 junior developers 
at my current company. Passionate about user experience and have side projects in fintech.
"""

print("=== JOB ANALYSIS ===")
analysis = assistant.analyze_job(sample_job_posting)
print(f"Role: {analysis.role_title}")
print(f"Experience Level: {analysis.experience_level}")
print(f"Key Requirements: {', '.join(analysis.key_requirements)}")
print(f"Strategy: {analysis.application_strategy}")

print("\n=== COVER LETTER STRUCTURE ===")
cover_letter = assistant.generate_cover_letter(analysis, candidate_background)
print(f"Opening Hook: {cover_letter.opening_hook}")
print(f"Tone: {cover_letter.tone_style}")

print("\n=== INTERVIEW PREP ===")
interview_prep = assistant.prepare_interview(analysis)
print(interview_prep)
