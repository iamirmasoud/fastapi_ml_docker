import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class UserInput(BaseModel):
    user_input: float


@app.get("/")
async def index():
    return {"Message": "This is Index"}
