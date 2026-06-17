import os
import io
import json
import asyncio

from functools import partial
import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import httpx


app = FastAPI()
load_dotenv("OPEN_KEY.env")

# assert os.getenv("OPEN_KEY") is not None

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_KEY"),
    timeout=60.0,
)


class JobExperience(BaseModel):
    company: str = None
    role: str = None
    startdatetime: Optional[str] = None
    enddatetime: Optional[str] = None
    description: str = None

class ProjectExperience(BaseModel):
    startdatetime: Optional[str] = None
    enddatetime: Optional[str] = None
    description: str = None

class EducationExperience(BaseModel):
    institution: str = None
    degree: str = None
    major: str = None
    minor: str = None
    startdatetime: Optional[str] = None
    enddatetime: Optional[str] = None

class Certification(BaseModel):
    institution: str = None
    name: str = None
    completeddatetime: Optional[str] = None
    expirationdatetime: Optional[str] = None

class ResumeSchema(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = []
    experience: List[JobExperience] = []
    education: List[EducationExperience] = []
    certifications: List[Certification] = []


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Pure sync function — safe to run in a thread executor."""
    raw_text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
    return raw_text


def _build_prompt(raw_text: str) -> str:
    return (
        "### Instruction: You are a resume data expert. "
        "Extract the following resume text into a JSON object with these fields: "
        "full_name (string), email (string), phone (string), "
        "skills (array of strings), "
        "experience (array of objects with: company, role, startdatetime, enddatetime, description), "
        "education (array of objects with: institution, degree, major, minor, startdatetime, enddatetime), "
        "certifications (array of objects with: institution, name, completeddatetime, expirationdatetime). "
        "Use only data present in the resume. Set fields to null if not found.\n\n"
        f"### Input:\n{raw_text}\n\n"
        "### Output:"
    )


def _call_local(raw_text: str) -> str:
    prompt = _build_prompt(raw_text)
    # Generous timeout: the model runs at ~6 t/s on CPU and resumes can produce
    # 400+ tokens, so a single inference pass can take well over a minute.
    resp = httpx.post(
        "http://localhost:8080/completion",
        json={
            "prompt": prompt,
            "max_tokens": 4096,
            "temperature": 0.1,
            "stop": ["### Instruction", "### Input"],
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    content = resp.json().get("content", "").strip()
    brace = content.find('{')
    if brace > 0:
        content = content[brace:]

    print(f"[LLM] len={len(content)}  preview={content[:120]!r}")
    if not content:
        raise ValueError("Model returned an empty response.")
    return content


@app.post("/pdf")
async def parse_resume(file: UploadFile = File(...)):
    print("New request: " + file.filename)
    try:
        pdf_bytes = await file.read()
        loop = asyncio.get_event_loop()
        raw_text = await loop.run_in_executor(
            None, _extract_text_from_pdf_bytes, pdf_bytes
        )
        print(f"[PDF] extracted {len(raw_text)} chars, preview: {raw_text[:200]!r}")
        with open("pdf.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)

        last_error = None
        for attempt in range(3):
            try:
                json_string = await loop.run_in_executor(
                    None, partial(_call_local, raw_text)
                )
            except ValueError as e:
                last_error = e
                print(f"[attempt {attempt+1}] LLM call failed: {e}")
                continue
            try:
                parsed_json = json.loads(json_string)
                resume = ResumeSchema(**parsed_json)
                if not resume.full_name and not resume.email and not resume.skills and not resume.experience:
                    raise ValueError("Model returned empty schema with no extracted data.")
                with open("difference.txt", "w", encoding="utf-8") as diff_f:
                    diff_f.write(remaining_text(json_string, "pdf.txt"))
                with open("pdf.json", "w", encoding="utf-8") as f:
                    json.dump(parsed_json, f, indent=4)
                return resume
            except Exception as e:
                last_error = e
                print(f"[attempt {attempt+1}] JSON parse/validation failed: {e}")

        raise HTTPException(
            status_code=500, detail=f"Failed after 3 attempts: {last_error}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def remaining_text(json_string: str = None, txt_path: str = None, coverage_threshold: float = 0.5) -> str:
    import re

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    def tokenize(text: str) -> list:
        return re.findall(r"[a-z0-9]+", text.lower())

    def extract_values(obj) -> list:
        values = []
        if isinstance(obj, dict):
            for v in obj.values():
                values.extend(extract_values(v))
        elif isinstance(obj, list):
            for item in obj:
                values.extend(extract_values(item))
        elif isinstance(obj, str) and obj.strip():
            values.append(obj.strip())
        return values

    parsed = json.loads(json_string)
    json_values = extract_values(parsed)

    known_tokens = set()
    for val in json_values:
        known_tokens.update(tokenize(val))

    uncovered_lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        line_tokens = tokenize(line)
        if not line_tokens:
            continue
        covered = sum(1 for t in line_tokens if t in known_tokens)
        coverage = covered / len(line_tokens)
        if coverage < coverage_threshold:
            uncovered_lines.append(line)

    return "\n".join(uncovered_lines)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=600)