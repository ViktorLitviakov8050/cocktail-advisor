content = """fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
jinja2==3.1.2
langchain==0.3.19
langchain-openai==0.3.6
langchain-community==0.3.18
faiss-cpu==1.7.4
pandas==2.1.3
python-dotenv==1.0.0
openai==1.63.2"""

with open('requirements.txt', 'w') as f:
    f.write(content) 