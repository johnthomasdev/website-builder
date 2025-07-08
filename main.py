from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import asyncio
from typing import Dict, Any, Optional
import uvicorn
import argparse
import traceback
import shutil
import zipfile
from agent import CodeAssistantAgent
from contextlib import asynccontextmanager
from embeddings import EmbeddingManager
from file_watcher import FileWatcher

# Initialize agent and other components
agent = CodeAssistantAgent()
embedding_manager = EmbeddingManager()
file_watcher = FileWatcher()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up application...")
    os.makedirs("generated_apps", exist_ok=True)
    
    # Initialize the agent and embedding manager
    await agent.initialize(embedding_manager)
    await embedding_manager.index_directory("generated_apps")
    
    # Start watching for file changes
    loop = asyncio.get_event_loop()
    file_watcher.start_watching(embedding_manager, loop)
    
    print("Application startup complete")
    yield
    # Shutdown
    print("Shutting down application...")
    file_watcher.stop_watching()
    await agent.shutdown()
    print("Application shutdown complete")

app = FastAPI(title="Local Code Assistant", lifespan=lifespan)

@app.post("/api/clear")
async def clear_session(session_id: str = "default"):
    """Clears the agent's memory, state, and all generated files."""
    try:
        print("--- Clearing project and state ---")
        # Reset the vector database
        await embedding_manager.reset()

        # First, shut down the agent to release the database file lock
        await agent.shutdown()

        # Delete the LangGraph state files
        db_files = ["langgraph_state.sqlite", "langgraph_state.sqlite-shm", "langgraph_state.sqlite-wal"]
        for f in db_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"Removed state file: {f}")
                except OSError as e:
                    print(f"Error removing file {f}: {e}")

        # Re-initialize the agent to create a new checkpointer and db files
        await agent.initialize(embedding_manager)
        
        # Delete all previously generated projects
        workspace_dir = "generated_apps"
        if os.path.isdir(workspace_dir):
            shutil.rmtree(workspace_dir)
            print(f"Removed directory: {workspace_dir}")
        os.makedirs(workspace_dir, exist_ok=True)
        print(f"Re-created directory: {workspace_dir}")

        print("--- Project cleared successfully ---")
        return {"status": "success", "message": f"Session '{session_id}' cleared and reset."}
    except Exception as e:
        print(f"Error clearing workspace: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------- New Download Endpoint ----------------------
# This endpoint bundles the requested generated project into a ZIP file
# and returns it so it can be downloaded by the client.

@app.get("/api/download/{project_name}")
async def download_project(project_name: str):
    """Create a ZIP archive of the project directory and return it."""
    project_dir = os.path.join("generated_apps", project_name)

    if not os.path.isdir(project_dir):
        raise HTTPException(status_code=404, detail="Project not found")

    # Path for the temporary zip file (overwrite if it already exists)
    zip_filename = f"{project_name}.zip"
    zip_path = os.path.join("generated_apps", zip_filename)

    try:
        # Create/overwrite the zip archive
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(project_dir):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    # Preserve the project folder structure inside the zip
                    arcname = os.path.relpath(file_path, project_dir)
                    zipf.write(file_path, arcname)
    except Exception as e:
        print(f"Error creating zip archive: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to create zip archive")

    # Return the created zip file
    return FileResponse(zip_path, media_type="application/zip", filename=zip_filename)

class ChatMessage(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    project_name: Optional[str] = None
    project_path: Optional[str] = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint"""
    try:
        result = await agent.process_message(message.message, message.session_id)
        return ChatResponse(**result)
    except Exception as e:
        print(f"Error processing chat message: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# This new endpoint will dynamically serve files from the 'generated_apps' directory.
# This avoids the caching issue of StaticFiles and ensures new files are always found.
@app.get("/generated/{project_name}/{file_path:path}")
async def serve_generated_file(project_name: str, file_path: str):
    """Serves a file from a specific generated project directory."""
    file_location = os.path.join("generated_apps", project_name, file_path)
    if os.path.exists(file_location):
        return FileResponse(file_location)
    print(f"File not found at: {file_location}")
    raise HTTPException(status_code=404, detail="File not found")

# Mount the frontend. This will serve the main UI and act as a catch-all.
# For any path that is not found, it will serve index.html.
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Local Code Assistant')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to run the server on')
    args = parser.parse_args()
    
    # Run the server with the specified port
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
