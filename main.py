from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Корневая страница с формой
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": None})

# Принимаем сообщение из формы
@app.post("/send", response_class=HTMLResponse)
async def send_message(request: Request, message: str = Form(...)):
    reply = f"Твое сообщение: '{message}'. Ну ты и молодец конечно!"
    return templates.TemplateResponse("index.html", {"request": request, "response": reply})
