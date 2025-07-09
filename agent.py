from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
    import sqlite3
    import warnings

    class _AsyncSqliteSaverFromSync(SqliteSaver):

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            try:
                self.conn.commit()
            finally:
                self.conn.close()
            return False

        @classmethod
        def from_conn_string(cls, conn_string: str):  # type: ignore[override]
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
from typing import Dict, List, Any, Optional, Annotated, TYPE_CHECKING
import json
import uuid
import os
import re
import aiosqlite
import traceback
from llm_client import LLMClient, ModelNotLoadedError
from typing import TypedDict
from embeddings import EmbeddingManager

if TYPE_CHECKING:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    thread_id: str
    current_project_path: Optional[str]
    generated_html: Optional[str]
    generated_css: Optional[str]
    generated_js: Optional[str]
    project_name: Optional[str]
    retrieved_context: Optional[str]


class CodeAssistantAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.memory_cm = None
        self.memory: Optional["AsyncSqliteSaver"] = None
        self.graph: Optional[Any] = None
        
    async def initialize(self, embedding_manager: EmbeddingManager):
        await self.llm_client.initialize()
        self.embedding_manager = embedding_manager
        
        self.memory_cm = AsyncSqliteSaver.from_conn_string("langgraph_state.sqlite")
        self.memory = await self.memory_cm.__aenter__()
        
        self.build_graph()

    async def shutdown(self):
        if self.memory_cm:
            await self.memory_cm.__aexit__(None, None, None)
        self.graph = None
        
    def build_graph(self):
        if self.graph is not None:
            return
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self.router_node)

        workflow.add_node("generate_html_from_scratch", self.generate_html_from_scratch_node)
        workflow.add_node("generate_css_from_scratch", self.generate_css_from_scratch_node)
        workflow.add_node("generate_js_from_scratch", self.generate_js_from_scratch_node)
        
        workflow.add_node("retrieve_context", self.retrieve_context_node)

        workflow.add_node("load_existing_project", self.load_existing_project_node)
        workflow.add_node("edit_html", self.edit_html_node)
        workflow.add_node("edit_css", self.edit_css_node)
        workflow.add_node("edit_js", self.edit_js_node)

        workflow.add_node("assemble_and_create", self.assemble_and_create_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            lambda state: "load_existing_project" if state.get("current_project_path") else "generate_html_from_scratch"
        )

        workflow.add_edge("generate_html_from_scratch", "generate_css_from_scratch")
        workflow.add_edge("generate_css_from_scratch", "generate_js_from_scratch")
        workflow.add_edge("generate_js_from_scratch", "assemble_and_create")
        
        workflow.add_edge("load_existing_project", "retrieve_context")
        workflow.add_edge("retrieve_context", "edit_html")
        workflow.add_edge("edit_html", "edit_css")
        workflow.add_edge("edit_css", "edit_js")
        workflow.add_edge("edit_js", "assemble_and_create")

        workflow.add_edge("assemble_and_create", END)
        
        self.graph = workflow.compile(checkpointer=self.memory)

    async def clear_session_state(self, session_id: str):
        if not self.memory:
            print("[WARN] Memory not initialized, cannot clear session state.")
            return
        
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
        if not self.graph:
            self.build_graph()
        assert self.graph is not None
        
        config = {"configurable": {"thread_id": session_id}}
        
        final_state = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=message)], "thread_id": session_id}, 
            config
        )
        
        last_message = final_state['messages'][-1]
        response = {
            "response": last_message.content if isinstance(last_message, AIMessage) else "I'm ready.",
            "project_name": final_state.get("project_name"),
            "project_path": final_state.get("current_project_path")
        }
        return response

    async def retrieve_context_node(self, state: AgentState) -> Dict[str, Any]:
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
        print("--- Router: Checking for existing project ---")
        if state.get("current_project_path"):
            print(f"Project '{state['current_project_path']}' is active. Preparing to edit.")
        else:
            print("No active project. Starting new project workflow.")
        return {}

    async def _generate_code(self, prompt: str, node_name: str) -> str:
        print(f"--- Running Node: {node_name} ---")
        code = await self.llm_client.generate(prompt, max_tokens=8192)
        return code.strip().replace("```html", "").replace("```css", "").replace("```javascript", "").replace("```", "").strip()

    async def generate_html_from_scratch_node(self, state: AgentState) -> Dict[str, Any]:
        user_message = state["messages"][-1].content
        prompt = rf'''You are an expert web developer. A user wants a website: "{user_message}".
Generate a complete `index.html` file from scratch that fulfills this request.
Return ONLY the raw HTML code. Do not include markdown formatting.'''
        generated_html = await self._generate_code(prompt, "generate_html_from_scratch")
        return {"generated_html": generated_html}

    async def generate_css_from_scratch_node(self, state: AgentState) -> Dict[str, Any]:
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
        project_path = state.get("current_project_path")
        if not project_path or not os.path.isdir(project_path):
            print(f"[ERROR] Project path '{project_path}' not found. Cannot edit.")
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
        user_message = state["messages"][-1].content
        existing_html = state.get("generated_html", "")
        retrieved_context = state.get("retrieved_context", "")
        
        prompt = f"""You are an expert web developer modifying an existing webpage.
**User's instruction:** "{user_message}"

**Relevant code snippets from the project (for context):**
```
{retrieved_context}
```

Your goal is to update the HTML below so it satisfies the user's request **while staying consistent** with any styles or scripts referenced in the context.

**Existing HTML (to be modified):**
```html
{existing_html}
```

Return ONLY the full, updated raw HTML. Do not include markdown formatting or explanations.
"""
        edited_html = await self._generate_code(prompt, "edit_html")
        return {"generated_html": edited_html}

    async def edit_css_node(self, state: AgentState) -> Dict[str, Any]:
        user_message = state["messages"][-1].content
        existing_css = state.get("generated_css", "")
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
        print("--- Assembling and creating project files ---")
        
        final_code = {
            "index.html": state.get("generated_html") or "<!-- HTML generation failed -->",
            "styles.css": state.get("generated_css") or "/* CSS generation failed */",
            "app.js": state.get("generated_js") or "// JavaScript generation failed",
        }

        project_name = "current_project"
        project_path = os.path.join("generated_apps", project_name)

        os.makedirs(project_path, exist_ok=True)
        for filename, content in final_code.items():
            with open(os.path.join(project_path, filename), 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"Project files created/updated at: {project_path}")

        if state.get("current_project_path"):
             response_msg = f"I have applied the updates to the project."
        else:
             response_msg = f"I've created a new project. You can see it in the preview."
       
        return {
            "messages": [AIMessage(content=response_msg)],
            "current_project_path": project_path,
            "project_name": project_name
        }