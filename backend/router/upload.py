from fastapi import UploadFile,APIRouter,File
from fastapi.responses import JSONResponse
from backend.controller.pdf_parser import parse_pdf,embed_chunks
from backend.controller.qdrant import VectorStore
app = APIRouter()

@app.post("/ingest")
async def upload_docs(file:UploadFile = File(...)):
    content= parse_pdf(file)
    
    embedded = embed_chunks(content)

    return JSONResponse({
        "message": "PDF processed",
        "text": content,
    })

@app.post("/search")
async def search_query(query:str):
    res= VectorStore().search(query=query)

    return JSONResponse({
        "message": "PDF processed",
        "res": res,
    })
    