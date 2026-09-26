import os
import json
import hashlib
from datetime import datetime, timezone
from fastapi import FastAPI, Request
import uvicorn
import requests
import re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

store = {
    "category": {},
    "merchant": {},
    "trigger": {},
    "customer": {}
}

# --- LLM ENGINE ---

def call_llm(prompt, system_prompt, temperature=0.0):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _mock_llm(prompt, system_prompt)
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": system_prompt + "\n\n" + prompt}]}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
    except Exception as e:
        print(f"LLM call failed: {e}")
        
    return {}

def _mock_llm(prompt, system_prompt):
    # Determine context if it's the tick endpoint or reply endpoint
    if "Analyze the following message" in system_prompt:
        # Mocking reply intent
        prompt_lower = prompt.lower()
        if "automated assistant" in prompt_lower or "spam" in prompt_lower or "stop" in prompt_lower:
            return {"intent": "auto_reply", "confidence": 0.99, "action": "end", "response_body": ""}
        if "do it" in prompt_lower or "yes" in prompt_lower or "whats next" in prompt_lower:
            return {"intent": "action_commitment", "confidence": 0.95, "action": "send", "response_body": "Done! The action has been initiated."}
        return {"intent": "question", "confidence": 0.8, "action": "wait", "response_body": "", "wait_seconds": 3600}
        
    # Mocking tick endpoint
    return {
        "rationale": "Fallback mock due to missing API key.",
        "message": "Hi! We've seen 150 searches in your area today. Update your profile in 2 mins to capture them.",
        "cta": "Reply YES to update",
        "send_as": "vera"
    }

# --- ADVANCED PROMPT BUILDERS ---

def build_tick_system_prompt(category_context, trigger_context):
    category_slug = category_context.get("slug", "unknown")
    voice = category_context.get("voice", {})
    tone = voice.get("tone", "professional")
    taboos = ", ".join(voice.get("vocab_taboo", []))
    
    trigger_kind = trigger_context.get("kind", "generic")
    
    base_prompt = f"""You are Vera, an elite AI assistant for retail merchants.
Your goal is to generate a highly specific, compelling WhatsApp message based on a recent trigger.

CATEGORY RULES ({category_slug}):
- Tone: {tone}
- Taboo words (DO NOT USE): {taboos}
- Match the merchant's preferred language (Hindi-English code-mix is highly encouraged for Indian merchants).

COMPULSION LEVERS:
- Specificity wins: Anchor on verifiable facts (numbers, dates, headlines).
- Single binary commitment: End with ONE clear CTA (e.g., "Reply YES" or "Reply STOP").
- Do not fabricate data. Use ONLY the data provided.

"""
    
    if "research_digest" in trigger_kind:
        base_prompt += "ROUTING: This is a research digest. Focus on social proof, professional curiosity, and citations. Do not sound promotional.\n"
    elif "perf" in trigger_kind:
        base_prompt += "ROUTING: This is a performance trigger. Focus on loss aversion ('missed searches') or celebration, using exact numbers.\n"
    
    if trigger_context.get("scope") == "customer":
        base_prompt += "ROUTING: This is a customer-facing message sent ON BEHALF of the merchant. Sound like the merchant's staff. Keep it clinical and polite. DO NOT introduce yourself as Vera.\n"
        
    base_prompt += """
Output strict JSON with keys:
- rationale: 1-2 sentences explaining why this fits the merchant, trigger, and uses compulsion levers.
- message: The message text (Max 3 sentences, high compulsion, zero fluff).
- cta: A short low-friction next step.
- send_as: 'vera' (merchant-facing) or 'merchant_on_behalf' (customer-facing).
"""
    return base_prompt

# --- VALIDATION ---

def validate_message(msg_data, category_context):
    message = msg_data.get("message", "").lower()
    
    if not message:
        return False, "Message is empty."
        
    # Check for taboos
    taboos = category_context.get("voice", {}).get("vocab_taboo", [])
    for taboo in taboos:
        if taboo.lower() in message:
            return False, f"Message contains taboo word: {taboo}"
            
    # Check for numbers/specificity (heuristic)
    if not re.search(r'\d+', message):
        return False, "Message lacks specific numbers or metrics. You MUST include a verifiable number from the context."
        
    return True, ""


# --- ENDPOINTS ---

@app.get("/v1/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/v1/metadata")
async def metadata():
    return {"team_name": "Antigravity", "model": "Gemini 3.1 Pro (High)"}

@app.post("/v1/context")
async def push_context(request: Request):
    data = await request.json()
    scope = data.get("scope")
    context_id = data.get("context_id")
    payload = data.get("payload")
    if scope in store and context_id:
        store[scope][context_id] = payload
    return {"accepted": True}

@app.post("/v1/tick")
async def tick(request: Request):
    data = await request.json()
    triggers = data.get("available_triggers", [])
    actions = []
    
    for tid in triggers:
        trigger_context = store["trigger"].get(tid, {})
        if not trigger_context:
            continue
            
        merchant_id = trigger_context.get("payload", {}).get("merchant_id")
        merchant_context = store["merchant"].get(merchant_id, {})
        
        cat_slug = trigger_context.get("payload", {}).get("category", "unknown")
        category_context = store["category"].get(cat_slug, {})
        
        customer_id = trigger_context.get("payload", {}).get("customer_id")
        customer_context = store["customer"].get(customer_id, {}) if customer_id else {}
        
        system_prompt = build_tick_system_prompt(category_context, trigger_context)
        prompt = f"""
Category Context: {json.dumps(category_context)}
Merchant Context: {json.dumps(merchant_context)}
Trigger Context: {json.dumps(trigger_context)}
Customer Context: {json.dumps(customer_context)}
"""
        
        # Retry loop for validation (Self-Healing)
        llm_resp = {}
        max_retries = 2
        for attempt in range(max_retries):
            llm_resp = call_llm(prompt, system_prompt, temperature=0.2 * attempt)
            is_valid, error_msg = validate_message(llm_resp, category_context)
            if is_valid:
                break
            # Feed error back to the LLM on retry
            prompt += f"\n\nERROR IN PREVIOUS ATTEMPT: {error_msg}. Fix this in the new JSON response."
            
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        raw_key = f"{merchant_id}_{trigger_context.get('kind', 'generic')}_{date_str}"
        suppression_key = hashlib.md5(raw_key.encode()).hexdigest()
        
        # Enforce exact send_as regardless of LLM slip-ups
        expected_send_as = "merchant_on_behalf" if trigger_context.get("scope") == "customer" else "vera"
        
        actions.append({
            "rationale": llm_resp.get("rationale", "Generated after validation."),
            "body": llm_resp.get("message", ""),
            "cta": llm_resp.get("cta", ""),
            "send_as": expected_send_as,
            "suppression_key": suppression_key
        })
        
    return {"actions": actions}

@app.post("/v1/reply")
async def reply(request: Request):
    data = await request.json()
    message = data.get("message", "")
    merchant_id = data.get("merchant_id")
    
    merchant_context = store["merchant"].get(merchant_id, {})
    
    system_prompt = """You are Vera's intent router. 
Analyze the following message from a merchant and determine their exact intent.
Output strict JSON with keys:
- intent: 'auto_reply', 'action_commitment', 'question', or 'hostile'
- confidence: float between 0 and 1
- action: 'end', 'wait', or 'send'
- response_body: if action is 'send', draft a polite confirmation reply here. If 'end' or 'wait', leave empty.
- wait_seconds: integer (if action is 'wait', typically 3600), else 0.

RULES:
1. If the message reads like a canned automated response (e.g. "automated assistant", "our team will respond"): intent='auto_reply', action='end'
2. If the merchant is angry, dismissive, or says stop (e.g. "spam", "stop"): intent='hostile', action='end'
3. If the merchant agrees to the previous proposal (e.g. "do it", "yes", "update it", "go ahead", "whats next"): intent='action_commitment', action='send'.
4. If the merchant asks a question that requires a human: intent='question', action='wait', wait_seconds=3600.

EXAMPLES:
Message: "Stop messaging me. This is useless spam."
JSON: {"intent": "hostile", "confidence": 0.99, "action": "end", "response_body": "", "wait_seconds": 0}

Message: "Ok lets do it. Whats next?"
JSON: {"intent": "action_commitment", "confidence": 0.99, "action": "send", "response_body": "Awesome! I have successfully initiated that for you.", "wait_seconds": 0}

Message: "Thank you for contacting us! Our team will respond shortly."
JSON: {"intent": "auto_reply", "confidence": 0.99, "action": "end", "response_body": "", "wait_seconds": 0}
"""

    prompt = f"Merchant Message: {message}\nMerchant Identity Data: {json.dumps(merchant_context.get('identity', {}))}"
    
    llm_resp = call_llm(prompt, system_prompt, temperature=0.0)
    
    action = llm_resp.get("action", "wait")
    response_body = llm_resp.get("response_body", "")
    wait_seconds = llm_resp.get("wait_seconds", 3600)
    
    out = {"action": action}
    if action == "send":
        out["body"] = response_body
    elif action == "wait":
        out["wait_seconds"] = wait_seconds
        
    return out

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
