import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [projectPath, setProjectPath] = useState(null); 
  const [showPreview, setShowPreview] = useState(false);
  const [previewVersion, setPreviewVersion] = useState(0);

  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        { role: 'assistant', content: 'Ask me to create a web app that...' }
      ]);
    }
  }, []);

  const sendMessage = async (message) => {
    if (!message.trim() || loading) return;

    const newMessages = [...messages, { role: 'user', content: message }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const { data } = await axios.post('/api/chat', {
        message,
        session_id: 'default'
      });
      
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      
      if (data.project_path) {
        setShowPreview(true);
        setProjectPath(data.project_path);
        setPreviewVersion(prev => prev + 1);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, there was an error processing your request.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleNewProject = async () => {
    setLoading(true);
    try {
      await axios.post('/api/clear', { session_id: 'default' });
      setMessages([
        { role: 'assistant', content: 'Ask me to create a web app that...' }
      ]);
      setShowPreview(false);
      setProjectPath(null);
      setInput('');
      console.log("New project started. Backend and frontend state cleared.");
    } catch (error) {
      console.error('Error clearing session:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, there was an error starting a new project.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!projectPath) return;
    try {
      const projectName = projectPath.split('/').pop(); 
      const response = await axios.get(`/api/download/${projectName}`, {
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${projectName}.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error downloading project:', error);
    }
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="app-container">
      <div className="main-content">
        <div className="header">
          <div className="header-icon"></div>
          <button className="download-project-button" onClick={handleDownload} disabled={loading || !showPreview}>Download</button>
          <button className="new-project-button" onClick={handleNewProject} disabled={loading}>Clear Project</button>
        </div>
        <h1>Idea to app in seconds.</h1>

        <div className="chat-history">
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              <div className="message-content">{m.content}</div>
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="message-content">Thinking…</div>
            </div>
          )}
        </div>

        <div className="input-container">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask me to create a web app that..."
            disabled={loading}
          />
          <div className="input-footer">
            <div className="footer-actions">
            </div>
            <div className="footer-right">
                <button className="send-button" onClick={() => sendMessage(input)} disabled={loading} title="Send Message">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                </button>
            </div>
          </div>
        </div>
      </div>

      {showPreview && (
        <div className="preview-container">
          <iframe
            key={previewVersion}
            src={`/generated/current_project/index.html`}
            title="Project Preview"
          />
        </div>
      )}
    </div>
  );
}

export default App;