document.addEventListener('DOMContentLoaded', () => {
    // Mode toggling logic
    const modeRadios = document.querySelectorAll('input[name="mode"]');
    const tutorSettings = document.getElementById('tutor-settings');
    const coachSettings = document.getElementById('coach-settings');
    const mcqSettings = document.getElementById('mcq-settings');
    const notesSettings = document.getElementById('notes-settings');
    const chatMessages = document.getElementById('chat-messages');

    let currentMode = 'auto';

    modeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            currentMode = e.target.value;
            // Clean settings container visibility
            tutorSettings.classList.remove('active');
            coachSettings.classList.remove('active');
            mcqSettings.classList.remove('active');
            notesSettings.classList.remove('active');
            
            if (currentMode === 'tutor') {
                tutorSettings.classList.add('active');
                addSystemMessage("Switched to <b>AI Tutor</b> mode. Ask me anything to retrieve context + generate an exam-ready answer.");
            } else if (currentMode === 'coach') {
                coachSettings.classList.add('active');
                addSystemMessage("Switched to <b>Study Coach</b> mode. Tell me what needs planning and I'll generate a realistic study schedule.");
            } else if (currentMode === 'mcq') {
                mcqSettings.classList.add('active');
                addSystemMessage("Switched to <b>MCQ Quiz Generator</b>. Give me a topic and I'll generate strict interactive MCQs based on your notes.");
            } else if (currentMode === 'notes') {
                notesSettings.classList.add('active');
                addSystemMessage("Switched to <b>Expert Note-Maker</b>. I will create beautiful handwritten-style study notes with visuals. Just type a topic!");

            } else {
                addSystemMessage("Switched to <b>⚡ Auto</b> mode. Just type naturally — I'll detect whether you want QA, Notes, Quiz, Study Plan, or Code and route automatically!");
            }
        });
    });

    // Auto-resize textarea
    const promptInput = document.getElementById('prompt-input');
    promptInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight < 150 ? this.scrollHeight : 150) + 'px';
    });

    // Chat Logic
    const chatForm = document.getElementById('chat-form');
    
    // Initial welcome message
    addSystemMessage("Welcome! The GTU RAG Pipeline is hot and connected to Groq. Select your persona on the left and start learning!");

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = promptInput.value.trim();
        if(!text) return;

        // Reset input
        promptInput.value = '';
        promptInput.style.height = 'auto';

        // Add User Message
        addUserMessage(text);

        // Extract settings
        const payload = {
            query: text,
            mode: currentMode,
            subject: document.getElementById('subject-input').value.trim() || undefined
        };

        if(currentMode === 'tutor') {
            payload.marks = parseInt(document.getElementById('marks-input').value);
        } else if(currentMode === 'coach') {
            payload.days = parseInt(document.getElementById('days-input').value);
            payload.hours_per_day = parseInt(document.getElementById('hours-input').value);
            payload.weak_areas = document.getElementById('weak-input').value.trim() || undefined;
        } else if(currentMode === 'mcq') {
            payload.total_questions = parseInt(document.getElementById('mcq-count').value);
            payload.difficulty = document.getElementById('mcq-diff').value;
        } else if(currentMode === 'notes') {
            payload.detail_level = document.getElementById('detail-level').value;
        }

        // Show typing indicator
        const typingId = addTypingIndicator();

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            removeElement(typingId);
            
            if(data.response) {
                // Show auto-detected intent if applicable
                if (data.detected_intent) {
                    const modeLabels = {
                        'tutor': '🎓 AI Tutor',
                        'coach': '📋 Study Coach',
                        'mcq': '📝 MCQ Quiz',
                        'notes': '📖 Note-Maker'
                    };
                    addSystemMessage(`Auto-detected intent: <b>${data.detected_intent}</b> → Routed to <b>${modeLabels[data.detected_mode] || data.detected_mode}</b>`);
                }
                
                const effectiveMode = data.detected_mode || currentMode;
                
                if (effectiveMode === 'mcq') {
                    try {
                        let jsonStr = data.response;
                        jsonStr = jsonStr.replace(/```json/g, '').replace(/```/g, '').trim();
                        const quizData = JSON.parse(jsonStr);
                        addQuizMessage(quizData);
                    } catch(jsonErr) {
                        addAiMessage("Failed to parse the MCQ JSON. Raw output:\n\n" + data.response, true);
                    }
                } else if (effectiveMode === 'notes') {
                    addAiMessage(data.response, false, true);
                } else {
                    addAiMessage(data.response);
                }
            } else {
                addAiMessage("Error: Did not receive a valid response format.", true);
            }
        } catch(err) {
            removeElement(typingId);
            addAiMessage("Network Error: " + err.message, true);
        }
    });

    function addUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'message user';
        div.innerHTML = `<div class="msg-bubble">${text}</div>`;
        chatMessages.appendChild(div);
        scrollToBottom();
    }
    function sanitizeMermaid(text) {
        // Fix common mistakes like A(Text) or A[Text] where Text has special chars
        // We look for patterns like NodeID(label) and NodeID[label]
        // This is a basic regex-based fix
        return text.replace(/([a-zA-Z0-9_-]+)\(([^)]+)\)/g, (match, id, label) => {
            if (label.includes('"')) return match; 
            return `${id}["${label}"]`;
        }).replace(/([a-zA-Z0-9_-]+)\[([^\]]+)\]/g, (match, id, label) => {
            if (label.includes('"')) return match;
            return `${id}["${label}"]`;
        });
    }

    function addAiMessage(markdownText, isError=false, isNotes=false) {
        const div = document.createElement('div');
        div.className = 'message ai';
        const renderedHTML = marked.parse(markdownText);

        if (isNotes) {
            const notesId = 'notes-' + Date.now();
            let notesHTML = `<div class="notes-bubble" id="${notesId}">`;
            notesHTML += `<div class="notes-download-bar">`;
            notesHTML += `<span>Study Notes</span>`;
            notesHTML += `<button class="download-btn" onclick="downloadNotes('${notesId}')">Download Image</button>`;
            notesHTML += `</div>`;
            notesHTML += renderedHTML;
            notesHTML += `</div>`;
            div.innerHTML = notesHTML;
        } else {
            div.innerHTML = `<div class="msg-bubble" ${isError? 'style="color:#ef4444;"':''}>${renderedHTML}</div>`;
        }

        // Convert markdown mermaid blocks into div.mermaid so parser can find them
        div.querySelectorAll('code.language-mermaid').forEach(el => {
            const pre = el.parentElement;
            const mermaidDiv = document.createElement('div');
            mermaidDiv.className = 'mermaid';
            // Sanitize the mermaid content before setting it
            mermaidDiv.textContent = sanitizeMermaid(el.textContent);
            pre.replaceWith(mermaidDiv);
        });

        chatMessages.appendChild(div);

        // Render Mermaid Diagrams if present
        if (window.mermaid) {
            window.mermaid.run({
                nodes: div.querySelectorAll('.mermaid')
            }).catch(e => console.error("Mermaid Render Error:", e));
        }

        scrollToBottom();
    }

    // Function to render the interactive Quiz UI
    function addQuizMessage(quizData) {
        const div = document.createElement('div');
        div.className = 'message ai';
        div.style.width = '100%';
        div.style.maxWidth = '100%';

        let html = `<div class="msg-bubble" style="width: 100%;">`;
        html += `<h2>Interactive MCQ Quiz: ${quizData.topic}</h2>`;
        html += `<p style="color: var(--text-muted); margin-bottom: 20px;">Subject: ${quizData.subject} (${quizData.estimated_time_minutes} min)</p>`;

        quizData.questions.forEach((q, i) => {
            html += `
            <div class="quiz-card" id="quiz-q-${i}">
                <div class="quiz-meta">Question ${i + 1} • ${q.difficulty} level</div>
                <div class="quiz-title">${q.question}</div>
                <div class="quiz-options">
            `;
            
            // Render the options
            ['A', 'B', 'C', 'D'].forEach(letter => {
                if(q.options[letter]) {
                    html += `<div class="quiz-option" data-letter="${letter}" data-correct="${q.correct_answer}" onclick="handleQuizClick(this, '${i}')">`;
                    html += `<strong>${letter}</strong>. &nbsp; ${q.options[letter]}`;
                    html += `</div>`;
                }
            });

            html += `</div>`;
            // Hidden Explanation
            html += `<div class="quiz-explanation" id="quiz-exp-${i}"><strong>Explanation:</strong> ${q.explanation}</div>`;
            html += `</div>`;
        });
        
        html += `</div>`;
        div.innerHTML = html;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    // Global function to handle clicks on the quiz dynamically
    window.handleQuizClick = function(element, qIndex) {
        // Find all options in this specific card
        const card = document.getElementById(`quiz-q-${qIndex}`);
        const options = card.querySelectorAll('.quiz-option');
        
        // Prevent multiple clicks by checking if explanation is already shown
        const exp = document.getElementById(`quiz-exp-${qIndex}`);
        if(exp.classList.contains('show')) return;

        const correctLetter = element.getAttribute('data-correct');
        const clickedLetter = element.getAttribute('data-letter');

        // Highlight correct options
        options.forEach(opt => {
            const letter = opt.getAttribute('data-letter');
            if(letter === correctLetter) {
                opt.classList.add('correct');
            } else if (letter === clickedLetter && letter !== correctLetter) {
                opt.classList.add('wrong');
            }
            opt.style.cursor = 'default';
        });

        // Reveal explanation
        exp.classList.add('show');
    };

    function addSystemMessage(text) {
        const div = document.createElement('div');
        div.className = 'message ai';
        div.innerHTML = `<div class="msg-bubble" style="opacity:0.8; font-style:italic;">🚀 ${text}</div>`;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const div = document.createElement('div');
        div.className = 'message ai';
        div.id = id;
        div.innerHTML = `
            <div class="msg-bubble typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        chatMessages.appendChild(div);
        scrollToBottom();
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if(el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Global function to download notes as image
    window.downloadNotes = function(notesId) {
        const notesElement = document.getElementById(notesId);
        if (!notesElement) return;

        // Hide the download bar for the screenshot
        const downloadBar = notesElement.querySelector('.notes-download-bar');
        if (downloadBar) downloadBar.style.display = 'none';

        // Generate image using html2canvas
        html2canvas(notesElement, {
            backgroundColor: '#fefefe',
            scale: 2,
            useCORS: true,
            logging: false
        }).then(canvas => {
            // Convert canvas to blob and download
            canvas.toBlob(blob => {
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = 'study-notes-' + Date.now() + '.png';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);

                // Restore download bar
                if (downloadBar) downloadBar.style.display = 'flex';
            }, 'image/png');
        }).catch(err => {
            console.error('Download error:', err);
            alert('Failed to generate image. Please try again.');
            if (downloadBar) downloadBar.style.display = 'flex';
        });
    };
});
