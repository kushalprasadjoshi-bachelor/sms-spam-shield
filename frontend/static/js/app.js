// Main application JavaScript for SMS Spam Shield

// Application state
const AppState = {
    selectedModels: new Set(['lr']),
    chatHistory: [],
    modelInfo: {},
    theme: 'light',
    
    // Initialize state from localStorage
    init() {
        // Load selected models
        const savedModels = localStorage.getItem('selectedModels');
        if (savedModels) {
            this.selectedModels = new Set(JSON.parse(savedModels));
        }
        
        // Load chat history
        const savedHistory = localStorage.getItem('chatHistory');
        if (savedHistory) {
            this.chatHistory = JSON.parse(savedHistory);
        }
        
        // Load theme
        this.theme = localStorage.getItem('theme') || 'light';
        this.applyTheme();
    },
    
    // Save state to localStorage
    save() {
        localStorage.setItem('selectedModels', JSON.stringify([...this.selectedModels]));
        localStorage.setItem('chatHistory', JSON.stringify(this.chatHistory));
        localStorage.setItem('theme', this.theme);
    },
    
    // Toggle theme
    toggleTheme() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        this.applyTheme();
        this.save();
    },
    
    // Apply theme to document
    applyTheme() {
        if (this.theme === 'dark') {
            document.documentElement.setAttribute('data-bs-theme', 'dark');
            document.getElementById('darkModeToggle').checked = true;
        } else {
            document.documentElement.removeAttribute('data-bs-theme');
            document.getElementById('darkModeToggle').checked = false;
        }
    },
    
    // Add message to chat history
    addMessage(message) {
        this.chatHistory.push(message);
        if (this.chatHistory.length > 50) {
            this.chatHistory = this.chatHistory.slice(-50);
        }
        this.save();
    },
    
    // Clear chat history
    clearHistory() {
        if (confirm('Are you sure you want to clear all history?')) {
            this.chatHistory = [];
            this.save();
            updateUI();
        }
    },
    
    // Get statistics
    getStats() {
        const total = this.chatHistory.filter(msg => msg.type === 'assistant').length;
        const spamCount = this.chatHistory.filter(msg => 
            msg.type === 'assistant' && 
            msg.result && 
            ['spam', 'phishing', 'scam'].includes(msg.result.category.toLowerCase())
        ).length;
        
        const confidences = this.chatHistory
            .filter(msg => msg.type === 'assistant' && msg.result)
            .map(msg => msg.result.confidence);
        
        const avgConfidence = confidences.length > 0 
            ? (confidences.reduce((a, b) => a + b, 0) / confidences.length) 
            : 0;
            
        return {
            total,
            spamCount,
            avgConfidence
        };
    }
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize application state
    AppState.init();
    
    // Set up event listeners
    document.getElementById('darkModeToggle').addEventListener('change', function() {
        AppState.toggleTheme();
    });
    
    // Add click handler to sample SMS button if exists
    const sampleBtn = document.getElementById('sampleSMSBtn');
    if (sampleBtn) {
        sampleBtn.addEventListener('click', function() {
            // Implementation for sample SMS modal
        });
    }
    
    // Initialize UI
    updateUI();
});

// Update UI based on application state
function updateUI() {
    // Update history display
    updateHistoryDisplay();
    
    // Update statistics
    updateStats();
    
    // Update model selection UI
    updateModelSelectionUI();
}

// Update history display
function updateHistoryDisplay() {
    const historyList = document.getElementById('historyList');
    const emptyHistory = document.getElementById('emptyHistory');
    
    if (!historyList) return;
    
    // Clear current display
    historyList.innerHTML = '';
    
    if (AppState.chatHistory.length === 0) {
        if (emptyHistory) {
            historyList.appendChild(emptyHistory);
            emptyHistory.classList.remove('d-none');
        }
        return;
    }
    
    if (emptyHistory) {
        emptyHistory.classList.add('d-none');
    }
    
    // Display last 10 history items
    const recentHistory = AppState.chatHistory.slice(-10).reverse();
    
    recentHistory.forEach((message, index) => {
        if (message.type === 'user') {
            const historyItem = createHistoryItem(message);
            historyList.appendChild(historyItem);
        }
    });
}

// Create a history item element
function createHistoryItem(message) {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.onclick = () => loadMessageDetails(message.id);
    
    // Truncate text
    const text = message.text.length > 100 
        ? message.text.substring(0, 100) + '...' 
        : message.text;
    
    // Find corresponding assistant message for category
    const assistantMsg = AppState.chatHistory.find(m => 
        m.id === message.id && m.type === 'assistant'
    );
    
    let category = 'Unknown';
    let categoryClass = 'secondary';
    
    if (assistantMsg && assistantMsg.result) {
        category = assistantMsg.result.category;
        categoryClass = assistantMsg.result.category_color || 'secondary';
    }
    
    div.innerHTML = `
        <div class="history-text">${escapeHtml(text)}</div>
        <div class="d-flex justify-content-between align-items-center">
            <span class="badge bg-${categoryClass}">${category}</span>
            <small class="text-muted">${message.time}</small>
        </div>
    `;
    
    return div;
}

// Update statistics display
function updateStats() {
    const stats = AppState.getStats();
    
    // Update total predictions
    const totalEl = document.getElementById('totalPredictions');
    if (totalEl) totalEl.textContent = stats.total;
    
    // Update spam count
    const spamEl = document.getElementById('spamCount');
    if (spamEl) spamEl.textContent = stats.spamCount;
    
    // Update average confidence
    const avgConfidenceEl = document.getElementById('avgConfidence');
    if (avgConfidenceEl) {
        avgConfidenceEl.textContent = (stats.avgConfidence * 100).toFixed(1) + '%';
    }
    
    // Update progress bars
    const predictionProgress = document.getElementById('predictionProgress');
    if (predictionProgress) {
        predictionProgress.style.width = `${Math.min(stats.total * 10, 100)}%`;
    }
    
    const spamProgress = document.getElementById('spamProgress');
    if (spamProgress) {
        const spamPercentage = stats.total > 0 ? (stats.spamCount / stats.total) * 100 : 0;
        spamProgress.style.width = `${spamPercentage}%`;
    }
    
    const confidenceProgress = document.getElementById('confidenceProgress');
    if (confidenceProgress) {
        confidenceProgress.style.width = `${stats.avgConfidence * 100}%`;
    }
}

// Update model selection UI
function updateModelSelectionUI() {
    // Update checkboxes
    AppState.selectedModels.forEach(model => {
        const checkbox = document.getElementById(`model-${model}`);
        if (checkbox) checkbox.checked = true;
        
        const card = document.getElementById(`modelCard-${model}`);
        if (card) card.classList.add('active');
    });
    
    // Update UI for unselected models
    ['lr', 'nb', 'svm', 'lstm'].forEach(model => {
        if (!AppState.selectedModels.has(model)) {
            const checkbox = document.getElementById(`model-${model}`);
            if (checkbox) checkbox.checked = false;
            
            const card = document.getElementById(`modelCard-${model}`);
            if (card) card.classList.remove('active');
        }
    });
}

// Toggle model selection
function toggleModel(modelId) {
    if (AppState.selectedModels.has(modelId)) {
        AppState.selectedModels.delete(modelId);
    } else {
        AppState.selectedModels.add(modelId);
    }
    
    // Update UI and save state
    updateModelSelectionUI();
    AppState.save();
    
    // Also update the corresponding checkbox in the model selector
    const checkbox = document.getElementById(`model-${modelId}-check`);
    if (checkbox) {
        checkbox.checked = AppState.selectedModels.has(modelId);
    }
}

// Select a single model (for dropdown)
function selectModel(modelId) {
    AppState.selectedModels.clear();
    AppState.selectedModels.add(modelId);
    updateModelSelectionUI();
    AppState.save();
}

// Load message details
function loadMessageDetails(messageId) {
    const message = AppState.chatHistory.find(m => m.id === messageId);
    if (message && message.type === 'user') {
        const smsInput = document.getElementById('smsInput');
        if (smsInput) {
            smsInput.value = message.text;
            
            // Scroll to input
            smsInput.scrollIntoView({ behavior: 'smooth' });
            smsInput.focus();
        }
    }
}

// Filter history by category
function filterHistory(category) {
    // Implementation for filtering history
    console.log(`Filter history by: ${category}`);
    // This would require updating the history display with filtered results
}

// Clear chat
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        // Keep only the welcome message
        const welcomeMessage = chatMessages.querySelector('.message.assistant');
        chatMessages.innerHTML = '';
        if (welcomeMessage) {
            chatMessages.appendChild(welcomeMessage);
        }
    }
    
    const smsInput = document.getElementById('smsInput');
    if (smsInput) {
        smsInput.value = '';
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Setup event listeners
function setupEventListeners() {
    // Model checkbox listeners
    ['lr', 'nb', 'svm', 'lstm'].forEach(model => {
        const checkbox = document.getElementById(`model-${model}`);
        if (checkbox) {
            checkbox.addEventListener('change', function() {
                toggleModel(model);
            });
        }
        
        const modelCheck = document.getElementById(`model-${model}-check`);
        if (modelCheck) {
            modelCheck.addEventListener('change', function() {
                toggleModel(model);
            });
        }
    });
    
    // SMS input auto-resize
    const smsInput = document.getElementById('smsInput');
    if (smsInput) {
        smsInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
    
    // Enter key to predict (Shift+Enter for new line)
    if (smsInput) {
        smsInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                predictSMS();
            }
        });
    }
}

// Export functions for use in other modules
window.toggleModel = toggleModel;
window.selectModel = selectModel;
window.clearChat = clearChat;
window.filterHistory = filterHistory;
window.loadMessageDetails = loadMessageDetails;

// Expose functions globally for templates
window.updateHistoryDisplay = updateHistoryDisplay;
window.updateModelSelectionUI = updateModelSelectionUI;
window.setupEventListeners = setupEventListeners;