def create_gitignore():
    content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Environment Variables
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/vector_store/
data/favorites.json

# Logs
*.log

# Docker
.docker/
docker-compose.override.yml
"""

    with open('.gitignore', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    create_gitignore() 