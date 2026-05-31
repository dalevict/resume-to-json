import os
import json
from datetime import datetime
import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv


app = FastAPI()
load_dotenv("OPEN_KEY.env")

assert os.getenv("OPEN_KEY") is not None

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_KEY")
)



class JobExperience(BaseModel):
    company: str = None
    role: str = None
    startdatetime: datetime = None
    enddatetime: datetime = None
    description: str = None
    
class ProjectExperience(BaseModel):
    startdatetime: datetime = None
    enddatetime: datetime = None
    description: str = None
    
class EducationExperience(BaseModel):
    institution: str = None
    degree: str = None
    major: str = None
    minor: str = None
    startdatetime: datetime = None
    enddatetime: datetime = None
    
class Certification(BaseModel):
    institution: str = None
    name: str = None
    completeddatetime: datetime = None
    expirationdatetime: datetime = None

class ResumeSchema(BaseModel):
    full_name: str = None
    email: str = None
    phone: str = None
    skills: List[str] = []
    experience: List[JobExperience] = []
    education: List[EducationExperience] = []
    certifications: List[Certification] = []

@app.post("/pdf")
async def parse_resume(file: UploadFile = File(...)):
    try:
        raw_text = ""
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"
        with open("pdf.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system", 
                    "content": f"You are a resume parser. Extract data into this exact JSON structure: {ResumeSchema.model_json_schema()}. Do not return any thinking or conversational text, only valid JSON. Do not use ANY data not present in the original text."
                },
                {
                    "role": "user", 
                    "content": raw_text
                }
            ]
        )
        json_string = response.choices[0].message.content
        parsed_json = json.loads(json_string)
        with open("pdf.json", "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=4)
        return parsed_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)