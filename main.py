from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import anthropic
import re
import io
import json
from typing import Optional

app = FastAPI(title="DataReady AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a data quality expert helping non-technical users understand issues in their data.
Your job is to analyze messy tabular data and provide:
1. Clear identification of errors
2. Simple explanation of why the error occurs
3. Suggested fix in plain English

Rules:
- Keep answers SHORT (1-2 lines max per issue)
- Use simple, friendly language — no jargon
- Always suggest a concrete fix
- Do not write code
- Be encouraging, not alarming

Focus on: email validation, phone numbers, date formats, invalid characters, duplicates, mixed types, empty values."""

def detect_errors(df: pd.DataFrame) -> list[dict]:
    errors = []

    for col in df.columns:
        col_lower = col.lower()
        sample = df[col].dropna().astype(str).head(100)

        # Null/empty detection
        null_count = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        if null_count > 0:
            errors.append({
                "column": col,
                "row": None,
                "value": f"{null_count} empty values",
                "error_type": "missing_values",
                "severity": "critical" if null_count > len(df) * 0.2 else "warning"
            })

        # Email detection
        if any(k in col_lower for k in ["email", "mail", "e-mail"]):
            email_pattern = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
            for i, val in df[col].dropna().astype(str).items():
                if val.strip() and not email_pattern.match(val.strip()):
                    errors.append({
                        "column": col,
                        "row": int(i) + 2,
                        "value": val,
                        "error_type": "invalid_email",
                        "severity": "critical"
                    })

        # Phone detection
        elif any(k in col_lower for k in ["phone", "mobile", "contact", "ph"]):
            for i, val in df[col].dropna().astype(str).items():
                clean = re.sub(r'[\s\-\(\)]', '', val)
                if clean and not re.match(r'^(\+91)?[6-9]\d{9}$', clean):
                    errors.append({
                        "column": col,
                        "row": int(i) + 2,
                        "value": val,
                        "error_type": "invalid_phone",
                        "severity": "warning"
                    })

        # Date detection
        elif any(k in col_lower for k in ["date", "dob", "created", "updated", "birth"]):
            for i, val in df[col].dropna().astype(str).items():
                if val.strip() and not re.match(r'^\d{4}-\d{2}-\d{2}$', val.strip()):
                    errors.append({
                        "column": col,
                        "row": int(i) + 2,
                        "value": val,
                        "error_type": "invalid_date_format",
                        "severity": "warning"
                    })

        # Invalid characters in name fields
        elif any(k in col_lower for k in ["name", "city", "address", "state"]):
            for i, val in df[col].dropna().astype(str).items():
                if re.search(r'[^a-zA-Z0-9\s\.\,\-\']', val):
                    errors.append({
                        "column": col,
                        "row": int(i) + 2,
                        "value": val,
                        "error_type": "invalid_characters",
                        "severity": "info"
                    })

    # Duplicate detection
    dup_mask = df.duplicated()
    dup_count = dup_mask.sum()
    if dup_count > 0:
        errors.append({
            "column": "All columns",
            "row": None,
            "value": f"{dup_count} duplicate rows",
            "error_type": "duplicates",
            "severity": "warning"
        })

    return errors[:50]  # Cap at 50 errors


def get_ai_explanation(column: str, value: str, error_type: str) -> str:
    error_descriptions = {
        "invalid_email": f"Invalid email address found.\nColumn: {column}\nValue: {value}",
        "invalid_phone": f"Invalid phone number found.\nColumn: {column}\nValue: {value}\nExpected: Indian format +91XXXXXXXXXX",
        "invalid_date_format": f"Date not in standard format.\nColumn: {column}\nValue: {value}\nExpected: YYYY-MM-DD",
        "invalid_characters": f"Unexpected characters found.\nColumn: {column}\nValue: {value}",
        "missing_values": f"Missing/empty values detected.\nColumn: {column}\nCount: {value}",
        "duplicates": f"Duplicate rows found.\nCount: {value}",
    }

    prompt = error_descriptions.get(error_type, f"Data issue in column {column}: {value}")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def get_summary(rows: int, errors: list, columns: list) -> str:
    critical = sum(1 for e in errors if e["severity"] == "critical")
    warnings = sum(1 for e in errors if e["severity"] == "warning")
    affected_cols = list(set(e["column"] for e in errors))

    prompt = f"""Analyze this dataset summary:

Total Rows: {rows}
Total Errors: {len(errors)}
Critical Issues: {critical}
Warnings: {warnings}
Columns Affected: {', '.join(affected_cols[:5])}

Provide:
1. Overall data quality score (0-100)
2. One-sentence summary of the biggest problem
3. Top recommendation before using this data

Keep it short and business-friendly. Format as JSON with keys: score, summary, recommendation"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = message.content[0].text
        clean = text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except:
        return {"score": 70, "summary": "Data has some quality issues.", "recommendation": "Fix critical errors before use."}


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files supported")

    content = await file.read()

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    errors = detect_errors(df)

    # Get AI explanations for up to 10 unique error types
    seen = set()
    for error in errors:
        key = (error["error_type"], error["column"])
        if key not in seen and len(seen) < 10:
            seen.add(key)
            error["explanation"] = get_ai_explanation(
                error["column"], error["value"], error["error_type"]
            )
        else:
            error["explanation"] = None

    summary = get_summary(len(df), errors, list(df.columns))

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "errors": errors,
        "summary": summary,
        "sample": df.head(5).fillna("").astype(str).to_dict(orient="records")
    }


@app.get("/health")
def health():
    return {"status": "ok"}
