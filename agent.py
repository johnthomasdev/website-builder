from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# The async SQLite checkpointer lives in a separate optional package (``langgraph-checkpoint-sqlite``).
# If that package is missing, importing it will raise ``ModuleNotFoundError`` at runtime.
# To keep the application usable even without the extra dependency, we gracefully fall back to the
# synchronous checkpointer that ships with the main ``langgraph`` package.
# NOTE: we alias the fallback to ``AsyncSqliteSaver`` so the rest of the code can remain unchanged.
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – Occurs when the optional extra isn't installed
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
    import sqlite3
    import warnings

    class _AsyncSqliteSaverFromSync(SqliteSaver):
        """Thin async wrapper around the synchronous ``SqliteSaver``.

        This lets the rest of the code (which awaits the context manager)
        keep working even if the async implementation is not available.
        All operations still execute synchronously because SQLite itself is
        blocking, but for a local development environment this is usually
        acceptable.
        """

        # The original ``SqliteSaver`` exposes only a *sync* context manager.
        # We add the async variations expected by the caller.

        async def __aenter__(self):  # noqa: D401 – keep signature simple
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: D401
            # Ensure connection is committed/closed the same way the parent
            # context manager would do.
            try:
                self.conn.commit()
            finally:
                self.conn.close()
            return False

        # Provide an async-friendly constructor mirroring the async saver.
        @classmethod
        def from_conn_string(cls, conn_string: str):  # type: ignore[override]
            # Keep the same signature but return *synchronously*; caller does
            # not await this method, only the subsequent __aenter__.
            conn = sqlite3.connect(conn_string, check_same_thread=False)
            return cls(conn)

    AsyncSqliteSaver = _AsyncSqliteSaverFromSync  # type: ignore[assignment]

    warnings.warn(
        "Optional dependency 'langgraph-checkpoint-sqlite' is not installed. "
        "Falling back to a synchronous wrapper. Install the extra with\n"
        "    pip install langgraph-checkpoint-sqlite\n"
        "for better async performance.",
        ImportWarning,
    )
from typing import Dict, List, Any, Optional, Annotated
import json
import uuid
import os
import re
import aiosqlite
import traceback
from llm_client import LLMClient, ModelNotLoadedError
from typing import TypedDict
from embeddings import EmbeddingManager

# Simplified AgentState for a scratch-first approach
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    thread_id: str
    # The path to the project being worked on.
    current_project_path: Optional[str]
    # Staging area for generated code
    generated_html: Optional[str]
    generated_css: Optional[str]
    generated_js: Optional[str]
    # The name of the project, for the UI
    project_name: Optional[str]
    # Retrieved context from the vector store
    retrieved_context: Optional[str]


class CodeAssistantAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.memory_cm = None
        # ``AsyncSqliteSaver`` may be substituted with the sync wrapper at runtime.
        # The annotation is for developer clarity; ignore static errors if the
        # symbol is not available under strict type checking.
        self.memory: Optional[AsyncSqliteSaver] = None  # type: ignore[name-defined]
        self.graph: Optional[Any] = None
        
    async def initialize(self, embedding_manager: EmbeddingManager):
        """Initialize agent and its components."""
        await self.llm_client.initialize()
        self.embedding_manager = embedding_manager
        
        self.memory_cm = AsyncSqliteSaver.from_conn_string("langgraph_state.sqlite")
        self.memory = await self.memory_cm.__aenter__()
        
        self.build_graph()

    async def shutdown(self):
        """Cleanly shutdown the agent's resources."""
        if self.memory_cm:
            await self.memory_cm.__aexit__(None, None, None)
        # Reset the graph to force a rebuild on next initialization
        self.graph = None
        
    def build_graph(self):
        """Builds the full agent workflow, including conversational editing."""
        if self.graph is not None:
            return
        workflow = StateGraph(AgentState)

        # Add a router to decide if we're creating a new project or editing an existing one
        workflow.add_node("router", self.router_node)

        # Nodes for generating from scratch
        workflow.add_node("generate_html_from_scratch", self.generate_html_from_scratch_node)
        workflow.add_node("generate_css_from_scratch", self.generate_css_from_scratch_node)
        workflow.add_node("generate_js_from_scratch", self.generate_js_from_scratch_node)
        
        # Node for the RAG step
        workflow.add_node("retrieve_context", self.retrieve_context_node)

        # Nodes for the editing workflow
        workflow.add_node("load_existing_project", self.load_existing_project_node)
        workflow.add_node("edit_html", self.edit_html_node)
        workflow.add_node("edit_css", self.edit_css_node)
        workflow.add_node("edit_js", self.edit_js_node)

        # Final node to assemble and create the project files
        workflow.add_node("assemble_and_create", self.assemble_and_create_node)

        # --- Build the graph connections ---
        workflow.set_entry_point("router")

        # The router decides where to go next
        workflow.add_conditional_edges(
            "router",
            # If a project is already active, go to the edit flow.
            lambda state: "load_existing_project" if state.get("current_project_path") else "generate_html_from_scratch"
        )

        # --- From-Scratch Generation Path ---
        workflow.add_edge("generate_html_from_scratch", "generate_css_from_scratch")
        workflow.add_edge("generate_css_from_scratch", "generate_js_from_scratch")
        workflow.add_edge("generate_js_from_scratch", "assemble_and_create")
        
        # --- Project Editing Path ---
        workflow.add_edge("load_existing_project", "retrieve_context") # RAG step
        workflow.add_edge("retrieve_context", "edit_html") # Then edit
        workflow.add_edge("edit_html", "edit_css")
        workflow.add_edge("edit_css", "edit_js")
        workflow.add_edge("edit_js", "assemble_and_create")

        # The final step before ending
        workflow.add_edge("assemble_and_create", END)
        
        self.graph = workflow.compile(checkpointer=self.memory)

    async def clear_session_state(self, session_id: str):
        """Clears the checkpoint history for a given session_id."""
        if not self.memory:
            print("[WARN] Memory not initialized, cannot clear session state.")
            return
        
        # This is a bit of a workaround as LangGraph's checkpointer doesn't
        # have a direct 'delete' method. We connect to the DB and delete the thread.
        try:
            conn = await aiosqlite.connect("langgraph_state.sqlite")
            cursor = await conn.cursor()
            await cursor.execute(
                "DELETE FROM threads WHERE thread_id = ?", (session_id,)
            )
            await conn.commit()
            await conn.close()
            print(f"Cleared state for session_id: {session_id}")
        except Exception as e:
            print(f"Error clearing session state from SQLite: {e}")
            traceback.print_exc()

    async def process_message(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """Process a user message using the stateful, conversational graph."""
        if not self.graph:
            self.build_graph()
        assert self.graph is not None
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Invoke the graph with the new message. The checkpointer handles loading/saving state.
        final_state = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=message)], "thread_id": session_id}, 
            config
        )
        
        # Extract the relevant information for the frontend from the final state.
        last_message = final_state['messages'][-1]
        response = {
            "response": last_message.content if isinstance(last_message, AIMessage) else "I'm ready.",
            "project_name": final_state.get("project_name"),
            "project_path": final_state.get("current_project_path")
        }
        return response

    async def retrieve_context_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieves relevant code snippets from the vector store."""
        print("--- Node: retrieve_context ---")
        user_message = state["messages"][-1].content
        
        if not self.embedding_manager:
            print("[WARN] Embedding manager not initialized. Skipping context retrieval.")
            return {"retrieved_context": ""}

        retrieved_docs = await self.embedding_manager.retrieve_similar(str(user_message), n_results=3)
        
        context_str = "\n".join(retrieved_docs)
        print(f"Retrieved context: {context_str[:300]}...")
        return {"retrieved_context": context_str}

    async def router_node(self, state: AgentState) -> Dict[str, Any]:
        """Determines whether to start a new project or edit an existing one."""
        print("--- Router: Checking for existing project ---")
        if state.get("current_project_path"):
            print(f"Project '{state['current_project_path']}' is active. Preparing to edit.")
        else:
            print("No active project. Starting new project workflow.")
        return {}

    async def _generate_code(self, prompt: str, node_name: str) -> str:
        """Helper function to call the LLM and handle errors."""
        print(f"--- Running Node: {node_name} ---")
        code = await self.llm_client.generate(prompt, max_tokens=8192)
        # We no longer expect JSON, so we can clean up the response
        return code.strip().replace("```html", "").replace("```css", "").replace("```javascript", "").replace("```", "").strip()

    async def generate_html_from_scratch_node(self, state: AgentState) -> Dict[str, Any]:
        """Generates HTML from scratch."""
        user_message = state["messages"][-1].content
        prompt = rf'''You are an expert web developer. A user wants a website: "{user_message}".
Generate a complete `index.html` file from scratch that fulfills this request.
Return ONLY the raw HTML code. Do not include markdown formatting.'''
        generated_html = await self._generate_code(prompt, "generate_html_from_scratch")
        return {"generated_html": generated_html}

    async def generate_css_from_scratch_node(self, state: AgentState) -> Dict[str, Any]:
        """Generates CSS from scratch based on the generated HTML."""
        user_message = state["messages"][-1].content
        generated_html = state.get("generated_html", "")
        prompt = rf'''You are an expert web developer creating a website for a user who wants: "{user_message}".
You have already generated the HTML. Now, write a complete `styles.css` file to style it appropriately.
**Generated HTML:**
```html
{generated_html}
```
Return ONLY the raw CSS code. Do not include markdown formatting.'''
        generated_css = await self._generate_code(prompt, "generate_css_from_scratch")
        return {"generated_css": generated_css}

    async def generate_js_from_scratch_node(self, state: AgentState) -> Dict[str, Any]:
        """Generates JavaScript from scratch for the new HTML and CSS."""
        user_message = state["messages"][-1].content
        generated_html = state.get("generated_html", "")
        generated_css = state.get("generated_css", "")
        prompt = rf'''You are an expert web developer creating a website for a user who wants: "{user_message}".
You have already generated the HTML and CSS. Now, create an `app.js` file to make it interactive.
**Generated HTML:**
```html
{generated_html}
```
**Generated CSS:**
```css
{generated_css}
```
Return ONLY the raw JavaScript code. Do not include markdown formatting.'''
        generated_js = await self._generate_code(prompt, "generate_js_from_scratch")
        return {"generated_js": generated_js}

    async def load_existing_project_node(self, state: AgentState) -> Dict[str, Any]:
        """Loads the files from the current project directory into the state."""
        project_path = state.get("current_project_path")
        if not project_path or not os.path.isdir(project_path):
            # This should ideally not happen if the router is correct
            print(f"[ERROR] Project path '{project_path}' not found. Cannot edit.")
            # We'll clear the path to force a new project flow next time.
            return {"current_project_path": None, "generated_html": None, "generated_css": None, "generated_js": None}
            
        print(f"--- Loading existing project from: {project_path} ---")
        
        def read_file_content(file_name):
            file_path = os.path.join(project_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return f"// {file_name} not found"

        project_name = os.path.basename(project_path)

        return {
            "project_name": project_name,
            "generated_html": read_file_content("index.html"),
            "generated_css": read_file_content("styles.css"),
            "generated_js": read_file_content("app.js"),
        }

    async def edit_html_node(self, state: AgentState) -> Dict[str, Any]:
        """Edits the HTML file based on user's new request."""
        user_message = state["messages"][-1].content
        existing_html = state.get("generated_html", "")
        
        prompt = f"""You are an expert web developer. A user wants to modify a webpage.
**User's instruction:** "{user_message}"

Your goal is to edit the following HTML to incorporate the user's request. Modify the code as needed.

**Existing HTML (to be modified):**
```html
{existing_html}
```

Return ONLY the new, full, raw HTML code. Do not include markdown formatting.
"""
        edited_html = await self._generate_code(prompt, "edit_html")
        return {"generated_html": edited_html}

    async def edit_css_node(self, state: AgentState) -> Dict[str, Any]:
        """Edits the CSS file based on the new HTML and the user's request."""
        user_message = state["messages"][-1].content
        existing_css = state.get("generated_css", "")
        # The HTML may have been edited in the previous step
        current_html = state.get("generated_html", "")
        retrieved_context = state.get("retrieved_context", "")
        
        prompt = f"""You are an expert web developer modifying a website.
**User's instruction:** "{user_message}"

**Relevant code snippets from the project (for context):**
```
{retrieved_context}
```

Your goal is to edit the following CSS to incorporate the user's request.
The CSS should style the provided HTML.

**Current HTML:**
```html
{current_html}
```

**Existing CSS (to be modified):**
```css
{existing_css}
```

Return ONLY the new, full, raw CSS code. Do not include markdown formatting.
"""
        edited_css = await self._generate_code(prompt, "edit_css")
        return {"generated_css": edited_css}


    async def edit_js_node(self, state: AgentState) -> Dict[str, Any]:
        """Edits the JS file based on the new HTML/CSS and the user's request."""
        user_message = state["messages"][-1].content
        existing_js = state.get("generated_js", "")
        current_html = state.get("generated_html", "")
        current_css = state.get("generated_css", "")
        retrieved_context = state.get("retrieved_context", "")

        prompt = f"""You are an expert web developer modifying a website.
**User's instruction:** "{user_message}"

**Relevant code snippets from the project (for context):**
```
{retrieved_context}
```

Your goal is to edit the following JavaScript to incorporate the user's request.
The script should work with the provided HTML and CSS.

**Current HTML:**
```html
{current_html}
```
**Current CSS:**
```css
{current_css}
```

**Existing JavaScript (to be modified):**
```javascript
{existing_js}
```

Return ONLY the new, full, raw JavaScript code. Do not include markdown formatting.
"""
        edited_js = await self._generate_code(prompt, "edit_js")
        return {"generated_js": edited_js}

    async def assemble_and_create_node(self, state: AgentState) -> Dict[str, Any]:
        """Assembles the final code, writes it to disk, and updates the state."""
        print("--- Assembling and creating project files ---")
        
        final_code = {
            "index.html": state.get("generated_html") or "<!-- HTML generation failed -->",
            "styles.css": state.get("generated_css") or "/* CSS generation failed */",
            "app.js": state.get("generated_js") or "// JavaScript generation failed",
        }

        # Use a fixed project name and path to simplify routing, as per user feedback.
        project_name = "current_project"
        project_path = os.path.join("generated_apps", project_name)

        # Create the files
        os.makedirs(project_path, exist_ok=True)
        for filename, content in final_code.items():
            with open(os.path.join(project_path, filename), 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"Project files created/updated at: {project_path}")

        # Formulate a clear response message
        if state.get("current_project_path"):
             response_msg = f"I have applied the updates to the project."
        else:
             response_msg = f"I've created a new project. You can see it in the preview."
       
        return {
            "messages": [AIMessage(content=response_msg)],
            "current_project_path": project_path,
            "project_name": project_name
        }