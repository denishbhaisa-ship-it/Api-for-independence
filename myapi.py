import os
import io
import json
import base64
import subprocess
import asyncio
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from PIL import Image
import requests
from bs4 import BeautifulSoup

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline
)
from threading import Thread

# =====================================================================
# 1. LOCAL ADVANCED AI MODEL LOAD (NO THIRD PARTY API)
# =====================================================================

# Pro Tip: Multimodal/Vision capabilities ke liye aap "Qwen/Qwen2-VL-2B-Instruct" use kar sakte hain
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 

print(f"[*] Initializing Enterprise Engine: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Hardware Accelerator: {device.upper()}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    print("[+] Base Model and Tokenizer successfully loaded into Memory.")
except Exception as e:
    print(f"[-] Model Loading Error: {e}")
    model, tokenizer = None, None

# =====================================================================
# 2. INTERNAL TOOL ARCHITECTURE (CLAUDE TOOL-USE SYSTEM)
# =====================================================================

class AgentTools:
    @staticmethod
    def execute_terminal(command: str) -> str:
        """Terminal, Shell aur Git commands run karta hai."""
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            if res.returncode == 0:
                return res.stdout if res.stdout else "Executed successfully."
            return f"Exit Code {res.returncode}: {res.stderr}"
        except Exception as e:
            return f"Execution Error: {str(e)}"

    @staticmethod
    def live_web_search(query: str) -> str:
        """Browser engine ke bina live deep web scraping karta hai."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            resp = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            snippets = [a.get_text(strip=True) for a in soup.find_all('a', class_='result__snippet', limit=4)]
            return "\n".join(snippets) if snippets else "No web results found."
        except Exception as e:
            return f"Web Search Failed: {str(e)}"

    @staticmethod
    def inspect_image(base64_str: str) -> str:
        """Image Base64 Decode & Vision Dimensions Processing."""
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            return f"Image successfully processed. Format: {image.format}, Size: {image.size}, Mode: {image.mode}"
        except Exception as e:
            return f"Image Processing Failed: {str(e)}"

# =====================================================================
# 3. CLAUDE-STYLE REQUEST & RESPONSE SCHEMAS
# =====================================================================

class ContentBlock(BaseModel):
    type: str  # "text" ya "image"
    text: Optional[str] = None
    image_base64: Optional[str] = None

class Message(BaseModel):
    role: str  # "user", "assistant", ya "system"
    content: Union[str, List[ContentBlock]]

class ClaudeStyleRequest(BaseModel):
    model: str = "custom-claude-core"
    system_prompt: Optional[str] = "You are an advanced enterprise AI assistant with tool capabilities."
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    enable_web_search: bool = False

# =====================================================================
# 4. FASTAPI ENGINE WITH STREAMING & TOOL PIPELINE
# =====================================================================

app = FastAPI(
    title="Custom Claude-Style Enterprise AI API",
    description="In-House Multimodal System Prompt Enabled Engine with Realtime Token Streaming & Tools"
)

def format_messages_to_prompt(system_prompt: str, messages: List[Message], web_data: str = "") -> str:
    """Claude format conversations ko Chat Template mein convert karta hai."""
    formatted_chat = []
    
    # System Instruction Inject
    full_system = system_prompt
    if web_data:
        full_system += f"\n\n[LIVE SEARCH RESULTS INJECTED]:\n{web_data}"
        
    formatted_chat.append({"role": "system", "content": full_system})
    
    for msg in messages:
        if isinstance(msg.content, str):
            formatted_chat.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            text_parts = []
            for block in msg.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "image" and block.image_base64:
                    img_info = AgentTools.inspect_image(block.image_base64)
                    text_parts.append(f"[IMAGE ATTACHMENT DETAILS: {img_info}]")
            formatted_chat.append({"role": msg.role, "content": "\n".join(text_parts)})
            
    return tokenizer.apply_chat_template(formatted_chat, tokenize=False, add_generation_prompt=True)

@app.post("/v1/messages")
async def generate_message(request: ClaudeStyleRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Local AI Model is not initialized.")

    # 1. Live Web Search handling
    web_data = ""
    if request.enable_web_search:
        last_user_msg = request.messages[-1].content
        query = last_user_msg if isinstance(last_user_msg, str) else "Web Search Request"
        web_data = AgentTools.live_web_search(query)

    # 2. Build Full Prompt
    prompt = format_messages_to_prompt(request.system_prompt, request.messages, web_data)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    # 3. REAL-TIME STREAMING RESPONSE (Claude Style Streaming)
    if request.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def stream_generator():
            for new_text in streamer:
                chunk = {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": new_text}
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # 4. STANDARD RESPONSE (Non-Streaming)
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "id": "msg_custom_claude_01",
            "type": "message",
            "role": "assistant",
            "model": request.model,
            "content": [
                {
                    "type": "text",
                    "text": response_text.strip()
                }
            ],
            "web_search_applied": request.enable_web_search
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")

# Endpoint 5: Direct Terminal / Shell Tool Execution
@app.post("/v1/tools/terminal")
async def run_cmd(command: str):
    return {"command": command, "output": AgentTools.execute_terminal(command)}

@app.get("/")
def root():
    return {
        "status": "Active",
        "engine": "Claude-Architected In-House AI Core",
        "features": ["System Prompts", "Vision Structure", "Streaming", "Live Web Search", "Tool Calling"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)