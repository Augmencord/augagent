import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const [prompt, setPrompt] = useState('')
  const [messages, setMessages] = useState([])
  const [connected, setConnected] = useState(false)
  const [apiKey, setApiKey] = useState('')
  const ws = useRef(null)

  const connect = () => {
    if (ws.current) ws.current.close()
    
    ws.current = new WebSocket(`ws://localhost:8000/stream?api_key=${apiKey}`)
    
    ws.current.onopen = () => {
      setConnected(true)
      setMessages(prev => [...prev, { role: 'system', content: 'Connected to AugAgent WebSocket.' }])
    }
    
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'chunk') {
        setMessages(prev => {
          const newMsgs = [...prev]
          const lastMsg = newMsgs[newMsgs.length - 1]
          if (lastMsg && lastMsg.role === 'assistant' && !lastMsg.final) {
            lastMsg.content += data.content
          } else {
            newMsgs.push({ role: 'assistant', content: data.content })
          }
          return newMsgs
        })
      } else if (data.type === 'result') {
        setMessages(prev => {
          const newMsgs = [...prev]
          newMsgs[newMsgs.length - 1].final = true
          return newMsgs
        })
      } else if (data.type === 'error') {
        setMessages(prev => [...prev, { role: 'error', content: data.message }])
      }
    }
    
    ws.current.onclose = () => setConnected(false)
  }

  const sendPrompt = () => {
    if (!ws.current || !connected) return
    setMessages(prev => [...prev, { role: 'user', content: prompt }])
    ws.current.send(JSON.stringify({ 
      prompt, 
      agent_config: { name: 'AugHomeAgent', role: 'Assistant', model: 'gpt-4o' } 
    }))
    setPrompt('')
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>AugHome IDE</h1>
        <div className="connection-panel">
          <input 
            type="password" 
            placeholder="API Key" 
            value={apiKey} 
            onChange={(e) => setApiKey(e.target.value)} 
          />
          <button onClick={connect}>{connected ? 'Reconnect' : 'Connect'}</button>
          <span className={`status ${connected ? 'connected' : 'disconnected'}`}></span>
        </div>
      </header>

      <main className="chat-area">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <span className="role-label">{m.role}</span>
            <pre className="content">{m.content}</pre>
          </div>
        ))}
      </main>

      <footer className="input-area">
        <textarea 
          value={prompt} 
          onChange={e => setPrompt(e.target.value)}
          placeholder="Ask AugAgent..."
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              sendPrompt();
            }
          }}
        />
        <button onClick={sendPrompt} disabled={!connected || !prompt.trim()}>Send</button>
      </footer>
    </div>
  )
}

export default App
