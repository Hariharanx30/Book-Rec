# run_livereload.py
import subprocess
import sys
from livereload import Server

# Start uvicorn using the same Python interpreter (so it uses the venv)
uvicorn_cmd = [
    sys.executable, "-m", "uvicorn", "app:app",
    "--host", "127.0.0.1", "--port", "8000"
]
proc = subprocess.Popen(uvicorn_cmd)

server = Server()
# Watch files you edit
server.watch('app.py')
server.watch('*.py')
server.watch('*.html')   # add templates/ or other dirs if you use them

try:
    print("Livereload running on http://127.0.0.1:35729 — open your app at http://127.0.0.1:8000")
    server.serve(port=35729, host='127.0.0.1', root='.')
finally:
    # ensure the uvicorn subprocess is terminated when server exits
    proc.terminate()
