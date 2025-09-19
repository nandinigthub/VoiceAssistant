from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.router.upload import app as upload_router
app = FastAPI(title="Voice Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

@app.get("/", include_in_schema=False)
async def read_root():
    return {"Hello": "World"}

app.include_router(upload_router)
