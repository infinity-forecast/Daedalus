document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('user-input');
    const messagesContainer = document.getElementById('messages');
    const chatContainer = document.getElementById('chat-container');
    const typingIndicator = document.getElementById('typing-indicator');
    const newChatBtn = document.getElementById('new-chat-btn');

    function scrollToBottom() {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    }

    function addMessage(content, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        
        const bubble = document.createElement('div');
        bubble.className = 'glass-bubble';
        
        // Simple markdown parsing for code blocks/bold
        // Replace **bold**
        let htmlContent = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Replace `code`
        htmlContent = htmlContent.replace(/`([^`]+)`/g, '<code>$1</code>');
        // Replace ```code blocks```
        htmlContent = htmlContent.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        // New lines
        htmlContent = htmlContent.replace(/\n/g, '<br>');

        bubble.innerHTML = htmlContent;
        
        msgDiv.appendChild(bubble);
        messagesContainer.appendChild(msgDiv);
        scrollToBottom();
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = input.value.trim();
        if (!text) return;

        // User message
        addMessage(text, 'user');
        input.value = '';
        input.blur(); // Hide keyboard on mobile

        // Show typing indicator
        typingIndicator.style.display = 'flex';
        chatContainer.appendChild(typingIndicator); // move to bottom
        scrollToBottom();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: text })
            });

            const data = await response.json();
            
            // Hide typing
            typingIndicator.style.display = 'none';
            
            if (response.ok) {
                addMessage(data.response, 'daedalus');
            } else {
                addMessage(`*Error: ${data.detail || 'Could not reach server'}*`, 'daedalus');
            }
        } catch (error) {
            console.error('Chat error:', error);
            typingIndicator.style.display = 'none';
            addMessage('*Network error. Please make sure the server is running.*', 'daedalus');
        }
    });

    newChatBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/new', { method: 'POST' });
            
            // Clear UI
            messagesContainer.innerHTML = '';
            addMessage("I am Daedalus. We start anew. What is on your mind?", 'daedalus');
            
        } catch (error) {
            console.error(error);
            addMessage('*Could not clear conversation.*', 'daedalus');
        }
    });
    
    // Initial resize to avoid iOS Safari issues
    window.addEventListener('resize', scrollToBottom);
});
