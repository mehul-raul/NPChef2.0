# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import os
# from dotenv import load_dotenv
# import openai
# from typing import Optional, List
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Initialize FastAPI app
# app = FastAPI(title="AI Recipe Assistant")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Setup templates
# templates = Jinja2Templates(directory="templates")

# # Configure OpenAI
# # openai.api_key = os.getenv("OPENAI_API_KEY")
# # if not openai.api_key:
# #     raise ValueError("OPENAI_API_KEY environment variable is not set")

# class RecipeRequest(BaseModel):
#     query: str = Field(..., min_length=1, description="The recipe to generate")
#     diet_preference: Optional[str] = Field(None, description="Dietary preference (e.g., vegetarian, vegan)")
#     cuisine_type: Optional[str] = Field(None, description="Type of cuisine (e.g., Italian, Mexican)")

#     class Config:
#         schema_extra = {
#             "example": {
#                 "query": "chocolate chip cookies",
#                 "diet_preference": "vegetarian",
#                 "cuisine_type": "italian"
#             }
#         }

# class LearningResource(BaseModel):
#     title: str
#     url: str
#     type: str

# class RecipeResponse(BaseModel):
#     recipe: str
#     image_url: str
#     learning_resources: List[LearningResource]

# # def generate_recipe(query: str, diet_preference: Optional[str] = None, cuisine_type: Optional[str] = None) -> dict:
# #     logger.info(f"Generating recipe for query: {query}, diet: {diet_preference}, cuisine: {cuisine_type}")
    
# #     if not query:
# #         raise HTTPException(status_code=400, detail="Recipe query is required")

# #     # Create a detailed prompt for the recipe
# #     prompt = f"""Create a detailed recipe for {query}"""
# #     if diet_preference:
# #         prompt += f" that is {diet_preference}"
# #     if cuisine_type:
# #         prompt += f" in {cuisine_type} style"
    
# #     prompt += """\n\nFormat the recipe in markdown with the following sections:
# #     1. Brief Description
# #     2. Ingredients (as a bulleted list)
# #     3. Instructions (as numbered steps)
# #     4. Tips (as a bulleted list)
# #     5. Nutritional Information (as a bulleted list)
    
# #     Use markdown formatting like:
# #     - Headers (###)
# #     - Bold text (**)
# #     - Lists (- and 1.)
# #     - Sections (>)
# #     """

# #     try:
# #         logger.info(f"Sending prompt to OpenAI: {prompt}")
        
# #         # Generate recipe text
# #         completion = openai.chat.completions.create(
# #             model="gpt-3.5-turbo",
# #             messages=[
# #                 {"role": "system", "content": "You are a professional chef who provides detailed recipes with ingredients, instructions, nutritional information, and cooking tips. Format your responses in markdown."},
# #                 {"role": "user", "content": prompt}
# #             ],
# #             temperature=0.7
# #         )
# #         recipe_text = completion.choices[0].message.content
# #         logger.info("Successfully generated recipe text")

# #         # Generate recipe image
# #         logger.info("Generating recipe image")
# #         image_response = openai.images.generate(
# #             model="dall-e-3",
# #             prompt=f"Professional food photography of {query}, appetizing, high-quality, restaurant style",
# #             n=1,
# #             size="1024x1024"
# #         )
# #         image_url = image_response.data[0].url
# #         logger.info("Successfully generated recipe image")

# #         # Get learning resources
# #         learning_resources = get_learning_resources(query)
# #         logger.info("Successfully generated learning resources")

# #         response_data = {
# #             "recipe": recipe_text,
# #             "image_url": image_url,
# #             "learning_resources": learning_resources
# #         }
        
# #         return response_data
# #     except Exception as e:
# #         logger.error(f"Error generating recipe: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))
# def generate_recipe(query: str, diet_preference: Optional[str] = None, cuisine_type: Optional[str] = None) -> dict:
#     logger.info(f"Generating mock recipe for query: {query}, diet: {diet_preference}, cuisine: {cuisine_type}")

#     mock_recipe = f"""
#     ### {query.title()} Recipe

#     > **A quick and easy mock recipe!**

#     #### Ingredients
#     - 1 cup flour
#     - 2 eggs
#     - 1/2 cup milk
#     - Salt to taste

#     #### Instructions
#     1. Mix all ingredients.
#     2. Cook on medium heat.
#     3. Serve hot.

#     #### Tips
#     - Use fresh ingredients.
#     - Adjust salt as per taste.

#     #### Nutritional Info
#     - Calories: ~200
#     - Protein: 5g
#     - Carbs: 30g
#     """

#     mock_image_url = "https://via.placeholder.com/600x400.png?text=Recipe+Image"

#     mock_learning_resources = [
#         {
#             "title": "Mock Cooking Basics",
#             "url": "https://example.com/mock-cooking",
#             "type": "video"
#         },
#         {
#             "title": "Mock Recipe Tips",
#             "url": "https://example.com/mock-tips",
#             "type": "article"
#         }
#     ]

#     return {
#         "recipe": mock_recipe,
#         "image_url": mock_image_url,
#         "learning_resources": mock_learning_resources
#     }

# def get_learning_resources(recipe_name: str) -> list:
#     return [
#         {
#             "title": f"Master the Art of {recipe_name}",
#             "url": f"https://cooking-school.example.com/learn/{recipe_name.lower().replace(' ', '-')}",
#             "type": "video"
#         },
#         {
#             "title": f"Tips and Tricks for Perfect {recipe_name}",
#             "url": f"https://recipes.example.com/tips/{recipe_name.lower().replace(' ', '-')}",
#             "type": "article"
#         }
#     ]

# @app.post("/recipe", response_model=RecipeResponse)
# async def get_recipe(request: RecipeRequest):
#     logger.info(f"Received recipe request: {request}")
#     try:
#         result = generate_recipe(request.query, request.diet_preference, request.cuisine_type)
#         logger.info("Successfully generated recipe response")
#         return result
#     except Exception as e:
#         logger.error(f"Error processing recipe request: {str(e)}")
#         return JSONResponse(
#             status_code=500,
#             content={"detail": str(e)}
#         )

# @app.get("/", response_class=HTMLResponse)
# async def root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080)
# import os
# import requests
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.responses import JSONResponse, HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# from typing import Optional
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# # Setup static + templates
# os.makedirs("static", exist_ok=True)
# os.makedirs("templates", exist_ok=True)

# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# # Hugging Face config
# TEXT_MODEL = "facebook/bart-large-cnn"
# HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# class RecipeRequest(BaseModel):
#     ingredients: str
#     diet: Optional[str] = None
#     cuisine: Optional[str] = None

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/api/generate-recipe")
# async def generate_recipe(request: RecipeRequest):
#     try:
#         # 👨‍🍳 Smart Prompt
#         prompt = f"""
#         You are a professional chef and recipe writer.
#         Create a full, detailed cooking recipe using the following ingredients: {request.ingredients}.
#         {f"Make sure it is suitable for a {request.diet} diet." if request.diet else ""}
#         {f"The recipe should follow {request.cuisine} cuisine style." if request.cuisine else ""}

#         Format the response in markdown with the following sections:
#         ### Title
#         ### Description
#         ### Ingredients (as a bulleted list)
#         ### Instructions (as numbered steps)
#         ### Tips
#         ### Nutritional Information (if possible)

#         Be friendly and helpful in tone.
#         """

#         # Debug logs
#         print("📤 Prompt Sent:", prompt)
#         print("🧠 Model:", TEXT_MODEL)

#         headers = {"Authorization": f"Bearer {HF_API_KEY}"}
#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": 250,
#                 "temperature": 0.8,
#                 "do_sample": True
#             }
#         }

#         # Send to Hugging Face
#         response = requests.post(
#             f"https://api-inference.huggingface.co/models/{TEXT_MODEL}",
#             headers=headers,
#             json=payload,
#             timeout=30
#         )

#         # Try JSON parse or show raw text
#         try:
#             result = response.json()
#         except Exception as e:
#             print("❌ Could not parse JSON. Raw response:")
#             print(response.text)
#             raise HTTPException(status_code=500, detail="Invalid response from Hugging Face API")

#         print("✅ HF JSON Response:", result)

#         # Handle errors or loading message
#         if "error" in result:
#             raise HTTPException(status_code=503, detail=result["error"])
#         if isinstance(result, dict) and "generated_text" in result:
#             generated = result["generated_text"]
#         elif isinstance(result, list) and "generated_text" in result[0]:
#             generated = result[0]["generated_text"]
#         else:
#             raise HTTPException(status_code=500, detail="No recipe generated by the model.")

#         return {
#             "recipe": generated.strip(),
#             "image_url": "/static/placeholder.jpg"
#         }

#     except Exception as e:
#         print(f"🔥 Error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

#uvicorn main:app --reload
import os
import shutil
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional
import torch

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# ========== Model & LLM Setup ==========
device = 0 if torch.cuda.is_available() else -1
print(f"🔥 Using device: {'GPU' if device==0 else 'CPU'}")

llm_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=512,
    temperature=0.7,
    device=device
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

DB_PATH = "vector_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ========== Prompt Templates ==========
non_rag_prompt = """
You are an expert Indian home chef and AI assistant.  
Generate a single, detailed, easy-to-follow recipe based only on the query.  
Output must be in **Markdown** format with clear sections:
- Ingredients
- Method
- Nutritional Info
- Cooking Tips

Be friendly and professional. Stop after the recipe.

Query: {query}
"""

rag_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert Indian home chef and AI assistant.

You are given some cooking knowledge from the user's personal recipe notes in <context>.
Use only the recipes from <context> that match the user's question.

STRICTLY FOLLOW MARKDOWN FORMAT.

✅ If the recipe exists in <context>, use it.
✅ If it needs improvement, create an improved version.
✅ If it does not exist, create a new one.
❌ DO NOT include unrelated recipes.

Always include: Ingredients, Method, Nutritional Info, Cooking Tips.

<context>
{context}
</context>

<user_question>
{question}
</user_question>

<response>
"""
)

# ========== Helper Functions ==========
def extract_non_rag_output(full_text: str) -> str:
    marker = "Ingredients:"
    idx = full_text.find(marker)
    return full_text[idx:].strip() if idx != -1 else full_text.strip()

def extract_rag_output(full_text: str) -> str:
    marker = "Generate a recipe for:"
    idx = full_text.find(marker)
    if idx != -1:
        return full_text[idx:].strip()
    marker2 = "Ingredients:"
    idx2 = full_text.find(marker2)
    return full_text[idx2:].strip() if idx2 != -1 else full_text.strip()

# ========== Routes ==========
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    print("✅ Serving index.html")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_recipe_file(file: UploadFile = File(...)):
    if file.filename.endswith(".txt") or file.filename.endswith(".pdf"):
        save_path = f"uploaded_files/{file.filename}"
        os.makedirs("uploaded_files", exist_ok=True)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = TextLoader(save_path)
        documents = loader.load()

        db = FAISS.from_documents(documents, embedding)
        db.save_local(DB_PATH)

        return {"message": "File uploaded and processed successfully."}
    else:
        return JSONResponse(status_code=400, content={"error": "Only .txt or .pdf files allowed."})

@app.post("/api/generate-recipe")  # NON-RAG
async def generate_recipe(
    ingredients: str = Form(...),
    diet: Optional[str] = Form("Any"),
    cuisine: Optional[str] = Form("Any")
):
    try:
        query = f"Give me a recipe using these ingredients: {ingredients}. Diet: {diet}, Cuisine: {cuisine}."
        response = llm.invoke(non_rag_prompt.format(query=query))
        cleaned_response = extract_non_rag_output(response)

        return {
            "recipe": cleaned_response,
            "image_url": "/static/placeholder.jpg"
        }

    except Exception as e:
        print("❌ Non-RAG failed:", e)
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

@app.post("/api/rag-recipe")  # RAG
async def rag_recipe(
    ingredients: str = Form(...),
    diet: Optional[str] = Form("Any"),
    cuisine: Optional[str] = Form("Any"),
    file: Optional[UploadFile] = File(None)
):
    try:
        query = f"Generate a recipe for: {ingredients}. Diet: {diet}. Cuisine style: {cuisine}."

        # Check if we have a file upload or need to use existing DB
        if file:
            extracted_text = ""
            if file.content_type == "application/pdf":
                pdf = PdfReader(file.file)
                extracted_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            elif file.content_type == "text/plain":
                extracted_text = (await file.read()).decode("utf-8")
            else:
                return JSONResponse(status_code=400, content={"detail": "Only .txt and .pdf supported"})

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            docs = text_splitter.create_documents([extracted_text])
            vector_store = FAISS.from_documents(docs, embedding)
        else:
            if not os.path.exists(DB_PATH):
                return JSONResponse(
                    status_code=400,
                    content={"detail": "No recipe database found. Please upload a file first."}
                )
            vector_store = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)

        retriever = vector_store.as_retriever(search_type="mmr", k=1)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": rag_prompt_template}
        )

        # Changed from "question" to "query" to match the prompt template
        result = qa_chain.invoke({"query": query})["result"]
        cleaned_result = extract_rag_output(result)

        return {
            "recipe": cleaned_result,
            "image_url": "/static/placeholder.jpg"
        }

    except Exception as e:
        print("❌ RAG failed:", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)


# uvicorn backend.main:app --reload --port 8008