from fastapi import FastAPI


######### Settings for logging ########
import os
import logging






app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


############################# TODO ############################
@app.post("/create")
async def create_dataset():
    pass

@app.get("/read")
async def read_data():
    pass

@app.post("/upload")
async def upload_data():
    pass

@app.post("/delete")
async def delete_data():
    pass

##################################TODO, pour MLOps pipeline ##############

@app.post("/loadDataset")
async def load_dataset_from_clearml():
    pass

@app.post("/run_mlops")
async def trigger_to_run_mlops():
    pass