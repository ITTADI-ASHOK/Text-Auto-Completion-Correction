import React, { useState } from "react";
import "./styles.css";

function App() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");

  const handleGenerate = async () => {
    const response = await fetch("http://localhost:5000/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input }),
    });
    const data = await response.json();
    setOutput(data.generated_code);
  };

  return (
    <div className="App">
      <h1>Code Auto-Correction and Generation</h1>
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter your code here..."
      />
      <button onClick={handleGenerate}>Generate</button>
      <pre>{output}</pre>
    </div>
  );
}

export default App;