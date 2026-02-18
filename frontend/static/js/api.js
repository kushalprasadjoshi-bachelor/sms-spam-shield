// API communication module for SMS Spam Shield

const API_BASE_URL = '/api/v1';

// API service object
const APIService = {
    // Health check
    async checkHealth() {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'unhealthy', error: error.message };
        }
    },
    
    // Get model information
    async getModelInfo() {
        try {
            const response = await fetch(`${API_BASE_URL}/models`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch model info:', error);
            return { models: {}, error: error.message };
        }
    },
    
    // Predict SMS
    async predictSMS(sms, models, includeExplanation = true, ensembleMethod = 'weighted_voting') {
        try {
            const modelList = Array.from(models || []);
            const payload = {
                sms: sms,
                models: modelList,
                include_explanation: includeExplanation
            };

            const endpoint = modelList.length > 1
                ? `${API_BASE_URL}/ensemble?method=${encodeURIComponent(ensembleMethod)}`
                : `${API_BASE_URL}/predict`;

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Prediction failed:', error);
            throw error;
        }
    },
    
    // Get sample SMS messages
    async getSampleMessages() {
        // This could be expanded to fetch from API if available
        return {
            samples: [
                {
                    text: "Congratulations! You won a $1000 Walmart gift card. Claim now: bit.ly/freegift",
                    category: "phishing",
                    description: "Phishing attempt with fake prize"
                },
                {
                    text: "URGENT: Your bank account has been compromised. Call now to verify: 1-800-SCAM",
                    category: "scam",
                    description: "Bank scam attempt"
                },
                {
                    text: "50% OFF all electronics this weekend only! Visit our store or shop online.",
                    category: "promotional",
                    description: "Promotional message"
                },
                {
                    text: "Hey, are we still meeting for lunch today at 1 PM?",
                    category: "legitimate",
                    description: "Legitimate personal message"
                }
            ]
        };
    },

    // Compare models
    async compareModels(sms) {
        try {
            const response = await fetch(`${API_BASE_URL}/compare`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ sms: sms })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Model comparison failed:', error);
            throw error;
        }
    },
    
    // Get ensemble methods
    async getEnsembleMethods() {
        try {
            const response = await fetch(`${API_BASE_URL}/ensemble/methods`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch ensemble methods:', error);
            return { methods: [], error: error.message };
        }
    }
};


// Load model information and update UI
async function loadModelInfo() {
    try {
        const data = await APIService.getModelInfo();
        
        if (data.models) {
            AppState.modelInfo = data.models;
            const loadedModels = Object.entries(data.models)
                .filter(([, model]) => model.status === 'loaded')
                .map(([modelKey]) => modelKey);

            if (loadedModels.length > 0) {
                AppState.selectedModels = new Set(
                    [...AppState.selectedModels].filter(model => loadedModels.includes(model))
                );
                if (AppState.selectedModels.size === 0) {
                    AppState.selectedModels.add(loadedModels[0]);
                }
                AppState.save();
                if (typeof updateModelSelectionUI === 'function') {
                    updateModelSelectionUI();
                }
            }
            updateModelMetrics();
            if (typeof updateModelTheoryFromSelection === 'function') {
                updateModelTheoryFromSelection();
            }
        }
        
        return data;
    } catch (error) {
        console.error('Error loading model info:', error);
        return null;
    }
}

// Update model metrics display
function updateModelMetrics() {
    const models = AppState.modelInfo;
    
    Object.keys(models).forEach(modelKey => {
        const model = models[modelKey];
        const metricsDiv = document.getElementById(`metrics-${modelKey}`);
        
        if (!metricsDiv) return;

        const accuracyEl = metricsDiv.querySelector('.metric-value');
        const f1El = metricsDiv.querySelectorAll('.metric-value')[1];
        const isLoaded = model.status === 'loaded';

        if (accuracyEl) {
            accuracyEl.textContent = isLoaded ? `${(model.accuracy * 100).toFixed(1)}%` : 'N/A';
        }
        if (f1El) {
            f1El.textContent = isLoaded ? `${(model.f1_score * 100).toFixed(1)}%` : 'N/A';
        }
    });
}

// Update model selection from UI
function updateModelSelection() {
    // Sync checkboxes with selected models
    ['lr', 'nb', 'svm', 'lstm'].forEach(model => {
        const checkbox = document.getElementById(`model-${model}-check`);
        if (checkbox) {
            checkbox.checked = AppState.selectedModels.has(model);
        }
    });
}

// Use a sample SMS
function useSample(sampleText) {
    const smsInput = document.getElementById('smsInput');
    if (smsInput) {
        smsInput.value = sampleText;
        smsInput.style.height = 'auto';
        smsInput.style.height = (smsInput.scrollHeight) + 'px';
        
        // Close modal if open
        const modal = bootstrap.Modal.getInstance(document.getElementById('samplesModal'));
        if (modal) modal.hide();
        
        // Focus on input
        smsInput.focus();
    }
}

// Show visualization modal
function showVisualization(messageId) {
    const message = AppState.chatHistory.find(m => m.id === messageId && m.type === 'assistant');
    if (!message || message.type !== 'assistant') return;
    
    // Create visualization data
    const visualizationData = prepareVisualizationData(message.result);
    
    // Update modal content
    updateVisualizationModal(visualizationData);
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('visualizationModal'));
    modal.show();
}

// Prepare visualization data
function prepareVisualizationData(result) {
    const data = {
        confidence: result.confidence,
        modelResults: result.models || [],
        tokens: result.explanation?.tokens || []
    };
    
    return data;
}

// Update visualization modal content
function updateVisualizationModal(data) {
    // Update confidence chart
    updateConfidenceChart(data);
    
    // Update comparison chart
    updateComparisonChart(data);
    
    // Update token importance
    updateTokenImportance(data.tokens);
}

// Update confidence chart
function updateConfidenceChart(data) {
    const ctx = document.getElementById('confidenceChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (window.confidenceChart) {
        window.confidenceChart.destroy();
    }
    
    const categories = ['Confidence', 'Uncertainty'];
    const values = [data.confidence * 100, (1 - data.confidence) * 100];
    const colors = ['#10b981', '#ef4444'];
    
    window.confidenceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: categories,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Update comparison chart
function updateComparisonChart(data) {
    const ctx = document.getElementById('visualComparisonChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (window.comparisonChart) {
        window.comparisonChart.destroy();
    }
    
    if (data.modelResults.length === 0) return;
    
    const modelNames = data.modelResults.map(m => m.name);
    const confidences = data.modelResults.map(m => m.confidence * 100);
    const colors = ['#3b82f6', '#8b5cf6', '#f59e0b', '#10b981'];
    
    window.comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Confidence (%)',
                data: confidences,
                backgroundColor: colors.slice(0, modelNames.length),
                borderColor: colors.slice(0, modelNames.length),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Confidence (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Confidence: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Update token importance display
function updateTokenImportance(tokens) {
    const container = document.getElementById('tokenImportance');
    if (!container) return;
    
    if (tokens.length === 0) {
        container.innerHTML = '<p class="text-muted text-center">No token importance data available</p>';
        return;
    }
    
    // Sort tokens by importance
    const sortedTokens = [...tokens].sort((a, b) => b.importance - a.importance);
    
    let html = '<div class="d-flex flex-wrap gap-2">';
    sortedTokens.forEach(token => {
        const importancePercent = (token.importance * 100).toFixed(1);
        const colorIntensity = Math.min(100, Math.max(20, token.importance * 100));
        
        html += `
            <div class="token-badge" style="
                background-color: rgba(239, 68, 68, ${colorIntensity / 100});
                padding: 0.5rem 1rem;
                border-radius: 20px;
                color: ${colorIntensity > 50 ? 'white' : 'black'};
                font-weight: bold;
                position: relative;
            ">
                ${token.word}
                <span class="importance-value" style="
                    font-size: 0.7rem;
                    position: absolute;
                    top: -8px;
                    right: -8px;
                    background: white;
                    border-radius: 50%;
                    width: 20px;
                    height: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border: 1px solid #ddd;
                ">
                    ${importancePercent}%
                </span>
            </div>
        `;
    });
    html += '</div>';
    
    container.innerHTML = html;
}

// Load ensemble methods and display in model
async function loadEnsembleMethods() {
    try {
        const data = await APIService.getEnsembleMethods();
        
        // Display methods in a modal or alert
        let methodsHtml = '<h6>Ensemble Methods</h6>';
        
        data.methods.forEach(method => {
            methodsHtml += `
                <div class="card mb-2">
                    <div class="card-body">
                        <h6>${method.name}</h6>
                        <p class="small text-muted">${method.description}</p>
                        <code>${method.formula}</code>
                    </div>
                </div>
            `;
        });
        
        // Show in modal
        const modalBody = `
            <div class="ensemble-methods">
                ${methodsHtml}
            </div>
        `;
        
        showCustomModal('Ensemble Methods', modalBody);
        
    } catch (error) {
        console.error('Error loading ensemble methods:', error);
    }
}

// Helper function to show custom modal
function showCustomModal(title, content) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('customModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'customModal';
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title"></h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body"></div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }
    
    // Update content
    modal.querySelector('.modal-title').textContent = title;
    modal.querySelector('.modal-body').innerHTML = content;
    
    // Show modal
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

// Export API functions
window.APIService = APIService;
window.loadModelInfo = loadModelInfo;
window.useSample = useSample;
window.showVisualization = showVisualization;
window.loadEnsembleMethods = loadEnsembleMethods;
