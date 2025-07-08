from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio
import os

from embeddings import EmbeddingManager

class CodeFileHandler(FileSystemEventHandler):
    def __init__(self, embedding_manager, loop):
        self.embedding_manager = embedding_manager
        self.loop = loop
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.html', '.css', '.js')):# type: ignore
            print(f"File modified: {event.src_path}")
            # Schedule the async task on the main event loop
            asyncio.run_coroutine_threadsafe(
                self.embedding_manager.index_file(event.src_path), 
                self.loop
            )

class FileWatcher:
    def __init__(self):
        self.observer = Observer()
        self.embedding_manager = None
        self.loop = None
        
    def start_watching(self, embedding_manager: EmbeddingManager, loop: asyncio.AbstractEventLoop):
        """Start watching the generated_apps directory"""
        self.embedding_manager = embedding_manager
        self.loop = loop
        
        handler = CodeFileHandler(self.embedding_manager, self.loop)
        self.observer.schedule(handler, "generated_apps", recursive=True)
        self.observer.start()
        print("File watcher started")
    
    def stop_watching(self):
        """Stop watching files"""
        self.observer.stop()
        self.observer.join()
        print("File watcher stopped")