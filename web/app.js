document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('user-input');
    const messagesContainer = document.getElementById('messages');
    const chatContainer = document.getElementById('chat-container');
    const typingIndicator = document.getElementById('typing-indicator');
    const newChatBtn = document.getElementById('new-chat-btn');

    // --- Diagnostic panel ---
    const diagPanel = document.getElementById('diagnostic-panel');
    const diagToggle = document.getElementById('diag-toggle');
    const diagBody = document.getElementById('diag-body');

    if (diagToggle) {
        diagToggle.addEventListener('click', () => {
            diagBody.classList.toggle('collapsed');
        });
    }

    async function fetchDiagnostic() {
        try {
            const res = await fetch('/api/diagnostic');
            if (!res.ok) return;
            const data = await res.json();
            updateDiagnosticUI(data);
        } catch (e) {
            // silently ignore
        }
    }

    function updateDiagnosticUI(data) {
        if (!data || data.error) return;

        const limbic = data.limbic || {};
        const grounding = data.grounding || {};
        const brainstem = data.brainstem || {};

        // Mood
        const moodEl = document.getElementById('diag-mood');
        if (moodEl) moodEl.textContent = limbic.mood || '--';

        // Dopamine [-1, 1] -> bar position
        const dopa = limbic.dopamine || 0;
        const dopaBar = document.getElementById('bar-dopamine');
        const dopaVal = document.getElementById('val-dopamine');
        if (dopaBar) {
            // Map [-1,1] to [0%,100%]: (dopa + 1) / 2 * 100
            const pct = ((dopa + 1) / 2) * 100;
            dopaBar.style.width = pct + '%';
        }
        if (dopaVal) dopaVal.textContent = dopa.toFixed(2);

        // Serotonin [0, 1]
        const sero = limbic.serotonin || 0;
        const seroBar = document.getElementById('bar-serotonin');
        const seroVal = document.getElementById('val-serotonin');
        if (seroBar) seroBar.style.width = (sero * 100) + '%';
        if (seroVal) seroVal.textContent = sero.toFixed(2);

        // Grounding [0, 1]
        const gScore = grounding.score;
        const gBar = document.getElementById('bar-grounding');
        const gVal = document.getElementById('val-grounding');
        if (gBar && gScore !== null && gScore !== undefined) {
            gBar.style.width = (gScore * 100) + '%';
            gBar.className = 'diag-bar-fill grounding';
            if (gScore < 0.3) gBar.classList.add('low');
            else if (gScore < 0.6) gBar.classList.add('mid');
        }
        if (gVal) gVal.textContent = gScore !== null && gScore !== undefined ? gScore.toFixed(2) : '--';

        // Self-loop [0, 1]
        const sLoop = grounding.self_loop;
        const slBar = document.getElementById('bar-selfloop');
        const slVal = document.getElementById('val-selfloop');
        if (slBar && sLoop !== null && sLoop !== undefined) {
            slBar.style.width = (sLoop * 100) + '%';
        }
        if (slVal) slVal.textContent = sLoop !== null && sLoop !== undefined ? sLoop.toFixed(2) : '--';

        // Crisis badge
        const crisisEl = document.getElementById('diag-crisis');
        if (crisisEl) {
            if (brainstem.crisis) {
                crisisEl.classList.remove('hidden');
            } else {
                crisisEl.classList.add('hidden');
            }
        }

        // Interaction count
        const intEl = document.getElementById('diag-interactions');
        if (intEl) intEl.textContent = (brainstem.interactions || 0) + ' interactions';
    }

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
        // Strip <think> blocks completely so visualization is entirely hidden
        htmlContent = htmlContent.replace(/<think>([\s\S]*?)<\/think>/g, '');
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
                fetchDiagnostic();
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

    // Fetch initial diagnostic state
    fetchDiagnostic();
});
