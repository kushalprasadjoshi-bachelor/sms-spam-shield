// Chat functionality for SMS Spam Shield

// Generate unique ID
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Format time
function formatTime(date = new Date()) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Add user message to chat
function addUserMessage(text) {
    const messageId = generateId();
    const message = {
        id: messageId,
        type: 'user',
        text: text,
        time: formatTime()
    };
    
    AppState.addMessage(message);
    appendMessageToChat(message);
    
    return messageId;
}

// Add assistant message to chat
function addAssistantMessage(messageId, result) {
    const message = {
        id: messageId,
        type: 'assistant',
        time: formatTime(),
        result: result
    };
    
    AppState.addMessage(message);
    appendMessageToChat(message);
    
    return message;
}

// Append message to chat UI
function appendMessageToChat(message) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    // Remove loading indicator if present
    const loadingIndicator = document.getElementById('loadingIndicator');
    if (loadingIndicator) {
        loadingIndicator.classList.add('d-none');
    }
    
    // Create message element
    const messageElement = createMessageElement(message);
    
    // Add to chat
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
        if (typeof setNavbarScrolledState === 'function') {
            setNavbarScrolledState();
        }
    }, 100);
}

// Create message element
function createMessageElement(message) {
    const div = document.createElement('div');
    div.className = `message ${message.type} animate-slide-in`;
    div.id = `message-${message.type}-${message.id}`;
    div.setAttribute('data-thread-id', message.id);
    
    if (message.type === 'user') {
        div.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <div class="message-sender">
                        <i class="fas fa-user text-primary"></i>
                        You
                    </div>
                    <div class="message-time">${message.time}</div>
                </div>
                <div class="message-text">
                    <p>${escapeHtml(message.text)}</p>
                </div>
            </div>
        `;
    } else {
        // Assistant message with prediction result
        const result = message.result;
        
        // Determine category badge color
        let categoryColor = 'secondary';
        const category = result.category ? result.category.toLowerCase() : '';
        
        if (category.includes('spam') || category.includes('scam')) {
            categoryColor = 'danger';
        } else if (category.includes('phishing')) {
            categoryColor = 'warning';
        } else if (category.includes('promotional')) {
            categoryColor = 'info';
        } else if (category.includes('legitimate')) {
            categoryColor = 'success';
        }
        
        // Format models display
        let modelsHtml = '';
        if (result.models && result.models.length > 0) {
            modelsHtml = `
                <div class="model-predictions mt-3">
                    <h6>Model Results:</h6>
                    <div class="row">
                        ${result.models.map(model => {
                            let modelColor = 'secondary';
                            if (model.name === 'Logistic Regression') modelColor = 'primary';
                            if (model.name === 'Naive Bayes') modelColor = 'secondary';
                            if (model.name === 'SVM') modelColor = 'warning';
                            if (model.name === 'LSTM') modelColor = 'success';
                            
                            return `
                                <div class="col-md-6 mb-2">
                                    <div class="card">
                                        <div class="card-body p-2">
                                            <div class="d-flex justify-content-between">
                                                <span class="badge bg-${modelColor}">
                                                    ${model.name}
                                                </span>
                                                <span class="fw-bold">${(model.confidence * 100).toFixed(1)}%</span>
                                            </div>
                                            <small class="text-muted">${model.prediction}</small>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }
        
        // Format explanation
        let explanationHtml = '';
        if (result.explanation && ((result.explanation.tokens || []).length > 0 || result.explanation.text)) {
            const tokensHtml = (result.explanation.tokens || [])
                .map(token => `
                    <span class="badge bg-info me-1 mb-1" title="Importance: ${(token.importance * 100).toFixed(1)}%">
                        ${token.word}
                    </span>
                `)
                .join('');
            
            explanationHtml = `
                <div class="explanation mt-3">
                    <h6>AI Explanation:</h6>
                    ${tokensHtml ? `<div class="explanation-tokens mb-2">${tokensHtml}</div>` : ''}
                    <p class="mt-2 small">${result.explanation.text || 'These words contributed most to the prediction.'}</p>
                </div>
            `;
        }
        
        div.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <div class="message-sender">
                        <i class="fas fa-robot text-primary"></i>
                        SMS Spam Shield
                    </div>
                    <div class="d-flex align-items-center gap-2">
                        <div class="message-time">${message.time}</div>
                        <button class="btn btn-sm btn-link text-danger p-0" title="Delete chat" onclick="deleteChat('${message.id}', event)">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="message-text">
                    <div class="prediction-result">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="prediction-category bg-${categoryColor}">
                                ${result.category || 'Unknown'}
                            </span>
                            <button class="btn btn-sm btn-outline-primary" onclick="showVisualization('${message.id}')">
                                <i class="fas fa-chart-bar"></i> View Details
                            </button>
                        </div>
                        
                        <div class="prediction-confidence">
                            <span>Confidence:</span>
                            <span class="fw-bold">${(result.confidence * 100).toFixed(1)}%</span>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                            </div>
                        </div>
                        
                        ${modelsHtml}
                        ${explanationHtml}
                        
                        <div class="mt-3 text-end">
                            <small class="text-muted">
                                Processed in ${result.processing_time || 0}ms using ${result.model_count || 0} model(s)
                            </small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    return div;
}

// Main prediction function
async function predictSMS() {
    const smsInput = document.getElementById('smsInput');
    const predictBtn = document.getElementById('predictBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    if (!smsInput || !smsInput.value.trim()) {
        showAlert('Please enter an SMS message to analyze.', 'warning');
        smsInput.focus();
        return;
    }
    
    if (AppState.selectedModels.size === 0) {
        showAlert('Please select at least one model.', 'warning');
        return;
    }
    
    const smsText = smsInput.value.trim();
    
    // Disable button and show loading
    if (predictBtn) {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    }
    
    if (loadingIndicator) {
        loadingIndicator.classList.remove('d-none');
    }
    
    try {
        // Add user message to chat
        const messageId = addUserMessage(smsText);
        
        // Get include explanation setting
        const includeExplanation = document.getElementById('includeExplanation')?.checked ?? true;
        const ensembleMethod = document.getElementById('ensembleMethodSelect')?.value || 'weighted_voting';
        
        // Call API
        const result = await APIService.predictSMS(
            smsText, 
            AppState.selectedModels, 
            includeExplanation,
            ensembleMethod
        );
        
        // Format result for display
        const formattedResult = formatPredictionResult(result);
        if (includeExplanation && !formattedResult.explanation) {
            formattedResult.explanation = {
                tokens: [],
                text: 'No token-level explanation was produced by the selected model(s) for this message.'
            };
        }
        
        // Add assistant message
        addAssistantMessage(messageId, formattedResult);
        
        // Clear input
        smsInput.value = '';
        smsInput.style.height = 'auto';
        
        // Update UI
        updateUI();
        
    } catch (error) {
        console.error('Prediction error:', error);
        
        // Show error message in chat
        const errorMessage = {
            id: generateId(),
            type: 'assistant',
            time: formatTime(),
            result: {
                category: 'Error',
                category_color: 'danger',
                confidence: 0,
                models: [],
                explanation: {
                    text: 'Failed to analyze SMS. Please try again.'
                },
                processing_time: 0,
                model_count: 0
            }
        };
        
        AppState.addMessage(errorMessage);
        appendMessageToChat(errorMessage);
        
        showAlert(`Analysis failed: ${error.message}`, 'danger');
        
    } finally {
        // Re-enable button and hide loading
        if (predictBtn) {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<i class="fas fa-robot"></i> Analyze SMS';
        }
        
        if (loadingIndicator) {
            loadingIndicator.classList.add('d-none');
        }
    }
}

// Format API prediction result for display
function formatPredictionResult(apiResult) {
    const result = {
        category: apiResult.ensemble_prediction || 'Unknown',
        confidence: apiResult.ensemble_confidence || 0,
        models: [],
        model_count: apiResult.individual_predictions?.length || 0,
        processing_time: apiResult.processing_time_ms || 0
    };
    
    // Format individual model predictions
    if (apiResult.individual_predictions) {
        result.models = apiResult.individual_predictions.map(pred => ({
            name: formatModelName(pred.model),
            prediction: pred.prediction || 'Unknown',
            confidence: pred.confidence || 0,
            color: getModelColor(pred.model)
        }));
    }
    
    // Add explanation if available
    if (apiResult.individual_predictions && apiResult.individual_predictions.length > 0) {
        const explainerPrediction = apiResult.individual_predictions.find(pred => pred.explanation);
        if (explainerPrediction && explainerPrediction.explanation) {
            const importantWords = explainerPrediction.explanation.important_tokens || [];
            const importanceMap = explainerPrediction.explanation.feature_importance || {};
            result.explanation = {
                method: explainerPrediction.explanation.method || 'unknown',
                tokens: importantWords.map((token, index) => ({
                    word: token,
                    importance: Math.max(
                        0.08,
                        Number(importanceMap[token] || 0) || Math.max(0.1, 0.9 - (index * 0.08))
                    )
                })),
                text: `Important words influencing the prediction (${explainerPrediction.model.toUpperCase()}):`
            };
        }
    }
    
    // Determine category color
    const category = result.category.toLowerCase();
    if (category.includes('spam') || category.includes('scam')) {
        result.category_color = 'danger';
    } else if (category.includes('phishing')) {
        result.category_color = 'warning';
    } else if (category.includes('promotional')) {
        result.category_color = 'info';
    } else if (category.includes('legitimate')) {
        result.category_color = 'success';
    } else {
        result.category_color = 'secondary';
    }
    
    return result;
}

// Format model name for display
function formatModelName(modelCode) {
    const modelNames = {
        'lr': 'Logistic Regression',
        'nb': 'Naive Bayes',
        'svm': 'SVM',
        'lstm': 'LSTM'
    };
    
    return modelNames[modelCode] || modelCode;
}

// Get color for model badge
function getModelColor(modelCode) {
    const colors = {
        'lr': 'primary',
        'nb': 'secondary',
        'svm': 'warning',
        'lstm': 'success'
    };
    
    return colors[modelCode] || 'secondary';
}

// Show alert message
function showAlert(message, type = 'info') {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show animate-slide-in`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page (you might want to add a specific container)
    const container = document.querySelector('.main-content');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Escape HTML to prevent XSS (redefined here for completeness)
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Export chat functions
window.predictSMS = predictSMS;
window.addUserMessage = addUserMessage;
window.addAssistantMessage = addAssistantMessage;
