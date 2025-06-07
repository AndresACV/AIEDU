from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "test-backend"}

@app.get("/")
async def root():
    return {"message": "Test server is working"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000) 