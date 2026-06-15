import os
import io
import json
import asyncio
from datetime import datetime
from functools import partial
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
    api_key=os.getenv("OPEN_KEY"),
    timeout=60.0,
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


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Pure sync function — safe to run in a thread executor."""
    raw_text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
    return raw_text


def _call_llm(messages: list) -> str:
    """Pure sync function — safe to run in a thread executor."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        response_format={"type": "json_object"},
        max_tokens=4096,
        messages=messages,
    )
    choice = response.choices[0]
    finish_reason = choice.finish_reason
    content = choice.message.content or ""

    print(f"[LLM] finish_reason={finish_reason!r}  len={len(content)}  preview={content[:120]!r}")

    if finish_reason == "length":
        # Model hit the token limit — response is truncated JSON, not parseable.
        # Raise so the retry loop can react with a corrective message.
        raise ValueError(
            f"Response truncated (finish_reason='length'). "
            f"Got {len(content)} chars. Consider a model with a larger output limit."
        )

    if not content.strip():
        raise ValueError("Model returned an empty response.")

    return content


@app.post("/pdf")
async def parse_resume(file: UploadFile = File(...)):
    try:
        # Read all bytes up front (async-safe)
        pdf_bytes = await file.read()

        # Offload blocking PDF parsing to a thread so the event loop stays free
        loop = asyncio.get_event_loop()
        raw_text = await loop.run_in_executor(
            None, _extract_text_from_pdf_bytes, pdf_bytes
        )

        with open("pdf.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a resume parser. Extract data into this exact JSON structure: "
                    f"{ResumeSchema().model_dump_json()}. "
                    "Do not return any thinking or conversational text, only valid JSON. "
                    "Do not use ANY data not present in the original text."
                ),
            },
            {"role": "user", "content": raw_text},
        ]

        last_error = None
        for attempt in range(3):
            try:
                # Offload blocking OpenAI call to a thread
                json_string = await loop.run_in_executor(
                    None, partial(_call_llm, messages)
                )
            except ValueError as e:
                # finish_reason='length' or empty response — log and retry
                last_error = e
                print(f"[attempt {attempt+1}] LLM call failed: {e}")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was unusable. "
                        "Return ONLY a compact valid JSON object matching the schema. "
                        "Omit all descriptions and details fields to reduce output size if needed."
                    ),
                })
                continue

            try:
                parsed_json = json.loads(json_string)
                resume = ResumeSchema(**parsed_json)
                print(remaining_text(json_string, "pdf.txt"))
                with open("pdf.json", "w", encoding="utf-8") as f:
                    json.dump(parsed_json, f, indent=4)
                return resume
            except Exception as e:
                last_error = e
                print(f"[attempt {attempt+1}] JSON parse/validation failed: {e}")
                messages.append({"role": "assistant", "content": json_string})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That response was invalid: {e}. "
                            "Return only valid JSON matching the schema, nothing else."
                        ),
                    }
                )

        raise HTTPException(
            status_code=500, detail=f"Failed after 3 attempts: {last_error}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def remaining_text(json_string: str = None, txt_path: str = None, coverage_threshold: float = 0.5) -> str:
    """
    Returns lines from the raw PDF text that are not SUBSTANTIALLY covered
    by the extracted JSON values.

    Strategy: token-level coverage.
      1. Collect all string leaf values from the JSON and tokenize them into
         a set of lowercase alphanumeric tokens (the "known" vocabulary).
      2. For every non-empty line in the raw PDF text, compute what fraction
         of its tokens appear in the known set.
      3. Lines whose coverage is below `coverage_threshold` are considered
         "not captured" and returned.

    This avoids the original substring-replace approach, which caused two bugs:
      - Partial matches: removing "Python" also stripped it from longer phrases.
      - LLM paraphrasing: the model rewrites bullet points, so the exact PDF
        text never matches and the whole section survives unreplaced.
    """
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

    # Build a flat set of all tokens present anywhere in the JSON
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
    uvicorn.run(app, host="0.0.0.0", port=8000)