from fastapi import FastAPI
app = FastAPI()

@app.get("/api/test")
def test():
    return {"message": "Python function is working on Vercel!"}
