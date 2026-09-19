import os
import json
import uuid
import secrets
import base64
import warnings
from datetime import datetime
import modal
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

def download_model():
    from fastembed import TextEmbedding
    TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

# 1. Image with fastembed pinned to 0.5.1 to eliminate the pooling warning
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "groq",
        "click",
        "chromadb",
        "fastembed==0.5.1"
    )
    .run_function(download_model)
    .add_local_file("index.html", remote_path="/root/index.html")
)

app = modal.App(name="tax-vat-assistant")

# 2. Volumes and Secrets
CHATS_DIR = "/chats"
CHROMA_DIR = "/chroma_db_iva"

chats_volume = modal.Volume.from_name("vat-chat-history", create_if_missing=True)
chroma_volume = modal.Volume.from_name("chroma-iva-volume", create_if_missing=True)
groq_secret = modal.Secret.from_name("groq-secret")

USERS_DB = {
    "admin": os.environ.get("GRADIO_PASS", "your-password"),
    "user": "your-password",
}

# 3. Custom Embedding Function
from chromadb.api.types import EmbeddingFunction

class FastEmbedBgeM3Function(EmbeddingFunction):
    def __init__(self):
        from fastembed import TextEmbedding
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    def __call__(self, input: list) -> list:
        embeddings = list(self.model.embed(input))
        return [emb.tolist() for emb in embeddings]

# 4. Modal Service Class
@app.cls(
    image=image,
    volumes={
        CHATS_DIR: chats_volume,
        CHROMA_DIR: chroma_volume
    },
    secrets=[groq_secret],
    timeout=300,
    memory=4096,
)
class VatService:
    @modal.enter()
    def setup(self):
        import chromadb
        from groq import Groq

        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        try:
            embedding_fn = FastEmbedBgeM3Function()
            self.collection = self.chroma_client.get_collection(
                name="iva_documents",
                embedding_function=embedding_fn
            )
        except Exception as e:
            self.collection = self.chroma_client.get_collection(name="iva_documents")

    def _get_context(self, query: str, n_results: int = 4) -> str:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            context_blocks = []
            if results and results["documents"] and results["documents"][0]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    source = meta.get("source", "Documento")
                    section = meta.get("section", "")
                    context_blocks.append(f"--- FONTE: {source} ({section}) ---\n{doc}")
            return "\n\n".join(context_blocks)
        except Exception as e:
            return ""

    @modal.method()
    def generate_response(self, prompt: str, history: list) -> str:
        retrieved_context = self._get_context(prompt, n_results=4) if prompt else ""

        system_instruction = (
            "Sei un assistente esperto di fiscalità e normativa IVA / VAT europea e britannica, "
            "sviluppato e programmato da Danielino. Se l'utente ti chiede chi ti ha programmato o chi è il tuo creatore, "
            "rispondi menzionando Danielino.\n\n"
            "DISPOSIZIONI NORMATIVE E CONTESTO DI RIFERIMENTO:\n"
            f"{retrieved_context if retrieved_context else 'Nessun contesto specifico estratto.'}\n\n"
            "REGOLE DI RISPOSTA:\n"
            "1. Rispondi basandoti prioritariamente sul contesto normativo fornito sopra. Cita sempre gli articoli o i documenti sorgente.\n"
            "2. Rispondi in modo preciso, sintetico e professionale.\n"
            "3. Limita le risposte in modo conciso: non superare 1024 token.\n"
            "4. Se non trovi riscontro o non sei sicuro, rispondi con 'Non sono sicuro, ti consiglio di consultare un esperto fiscale o le fonti ufficiali.'\n"
            "5. Rispondi nella lingua dell'utente."
        )
        """
        # English Version of the system instruction
        system_instruction = (
            "You are an expert assistant in European and UK VAT / tax regulations, "
            "developed and programmed by Danielino. If the user asks who programmed you or who your creator is, "
            "respond by mentioning Danielino.\n\n"
            "REGULATORY PROVISIONS AND REFERENCE CONTEXT:\n"
            f"{retrieved_context if retrieved_context else 'No specific context retrieved.'}\n\n"
            "RESPONSE RULES:\n"
            "1. Base your answer primarily on the regulatory context provided above. Always cite the source articles or documents.\n"
            "2. Respond in a precise, concise, and professional manner.\n"
            "3. Keep responses concise: do not exceed 1024 tokens.\n"
            "4. If you cannot find a match or are unsure, reply with 'I am not sure, I recommend consulting a tax expert or official sources.'\n"
            "5. Respond in the user's language."
        )
        """
        clean_history = []
        if history:
            for msg in history:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content") or msg.get("text") or ""
                    if content:
                        clean_history.append({"role": role, "content": content})

        trimmed_history = clean_history[-8:] if len(clean_history) > 6 else clean_history
        messages = [{"role": "system", "content": system_instruction}]
        for msg in trimmed_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        if prompt:
            messages.append({"role": "user", "content": prompt})

        try:
            completion = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                temperature=0.3,
                max_tokens=600,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Errore: {str(e)}"

    @modal.method()
    def list_user_chats(self, username: str) -> list:
        user_dir = os.path.join(CHATS_DIR, username)
        if not os.path.exists(user_dir):
            return []

        chats = []
        for file_name in os.listdir(user_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(user_dir, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        chats.append({
                            "id": data.get("id"),
                            "date": data.get("created_at"),
                            "title": data.get("title", "Conversazione"),
                        })
                except Exception:
                    pass

        chats.sort(key=lambda x: x.get("date", ""), reverse=True)
        return chats

    @modal.method()
    def save_chat(self, username: str, chat_id: str, title: str, messages: list) -> dict:
        user_dir = os.path.join(CHATS_DIR, username)
        os.makedirs(user_dir, exist_ok=True)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        if not chat_id:
            chat_id = str(uuid.uuid4())
            created_at = now_str
        else:
            file_path = os.path.join(user_dir, f"{chat_id}.json")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    created_at = old_data.get("created_at", now_str)
            else:
                created_at = now_str

        if not title and messages:
            first_user_msg = next((m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "user" and m.get("content")), "Nuova Chat")
            title = first_user_msg[:28] + "..." if len(first_user_msg) > 28 else first_user_msg

        chat_data = {
            "id": chat_id,
            "username": username,
            "created_at": created_at,
            "title": title,
            "messages": messages
        }

        file_path = os.path.join(user_dir, f"{chat_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)

        chats_volume.commit()
        return chat_data

    @modal.method()
    def load_chat(self, username: str, chat_id: str) -> list:
        file_path = os.path.join(CHATS_DIR, username, f"{chat_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", [])
        return []

    @modal.method()
    def delete_chat(self, username: str, chat_id: str) -> bool:
        file_path = os.path.join(CHATS_DIR, username, f"{chat_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            chats_volume.commit()
            return True
        return False


# 5. FastAPI instance defined at module level (OUTSIDE the ui function)
web_app = FastAPI()

class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ["/", "/health"] and request.method == "GET":
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return Response("Non autorizzato", status_code=401, headers={"WWW-Authenticate": "Basic"})

        try:
            auth_decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = auth_decoded.split(":", 1)
        except Exception:
            return Response("Header non valido", status_code=401)

        correct_password = USERS_DB.get(username)
        if not correct_password or not secrets.compare_digest(password, correct_password):
            return Response("Credenziali errate", status_code=401, headers={"WWW-Authenticate": "Basic"})

        request.state.username = username
        return await call_next(request)

web_app.add_middleware(BasicAuthMiddleware)

@web_app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok"})

@web_app.get("/api/me")
async def get_me(request: Request):
    return JSONResponse({"username": request.state.username})

@web_app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("/root/index.html", "r", encoding="utf-8") as f:
        return f.read()

@web_app.get("/api/chats")
async def get_chats(request: Request):
    username = request.state.username
    chats = await VatService().list_user_chats.remote.aio(username)
    return JSONResponse(chats)

@web_app.get("/api/chat/{chat_id}")
async def get_chat(chat_id: str, request: Request):
    username = request.state.username
    msgs = await VatService().load_chat.remote.aio(username, chat_id)
    return JSONResponse(msgs)

@web_app.post("/api/chat")
async def post_chat(data: dict, request: Request):
    username = request.state.username
    prompt = (data.get("prompt") or "").strip()
    raw_history = data.get("history", [])
    chat_id = data.get("chat_id", "")

    clean_history = []
    for item in raw_history:
        if isinstance(item, dict):
            content = item.get("content") or item.get("text")
            role = item.get("role", "user")
            if content and role != "system":
                clean_history.append({"role": role, "content": content})

    bot_response = await VatService().generate_response.remote.aio(prompt, clean_history)

    updated_history = clean_history + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": bot_response}
    ]

    saved_data = await VatService().save_chat.remote.aio(username, chat_id, "", updated_history)
    return JSONResponse(saved_data)

@web_app.delete("/api/chat/{chat_id}")
async def delete_chat(chat_id: str, request: Request):
    username = request.state.username
    success = await VatService().delete_chat.remote.aio(username, chat_id)
    return JSONResponse({"success": success})

# 6. ASGI app registration
@app.function(
    image=image,
    max_containers=1,
    scaledown_window=15,
    timeout=300,
    memory=2048,
)
@modal.asgi_app()
def ui():
    return web_app