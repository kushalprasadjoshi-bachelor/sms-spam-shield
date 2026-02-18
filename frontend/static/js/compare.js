// Model comparison functionality for SMS Spam Shield

class ModelComparison {
    constructor() {
        this.comparisonData = null;
        this.comparisonChart = null;
    }
    
    // Compare models for given SMS
    async compareModels(smsText) {
        try {
            const response = await APIService.compareModels(smsText);
            this.comparisonData = response;
            this.updateComparisonUI();
            this.createComparisonChart();
            return response;
        } catch (error) {
            console.error('Model comparison failed:', error);
            showAlert(`Comparison failed: ${error.message}`, 'danger');
            return null;
        }
    }
    
    // Update comparison UI
    updateComparisonUI() {
        if (!this.comparisonData) return;
        
        const comparisonContainer = document.getElementById('comparisonContainer');
        if (!comparisonContainer) return;
        
        const { comparison, summary } = this.comparisonData;
        
        let html = `
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">
                        <i class="fas fa-balance-scale-left me-2"></i>
                        Model Comparison Results
                    </h5>
                </div>
                <div class="card-body">
                    <div class="row mb-4">
                        <div class="col-md-4">
                            <div class="stat-card text-center p-3 bg-light rounded">
                                <div class="stat-value display-6">${summary.total_models}</div>
                                <div class="stat-label">Total Models</div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="stat-card text-center p-3 bg-light rounded">
                                <div class="stat-value display-6">${summary.successful_models}</div>
                                <div class="stat-label">Successful</div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="stat-card text-center p-3 bg-light rounded">
                                <div class="stat-value display-6">
                                    ${summary.agreement ? 
                                        '<i class="fas fa-check text-success"></i>' : 
                                        '<i class="fas fa-times text-danger"></i>'}
                                </div>
                                <div class="stat-label">Agreement</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Prediction</th>
                                    <th>Confidence</th>
                                    <th>Status</th>
                                    <th>Details</th>
                                </tr>
                            </thead>
                            <tbody>
        `;
        
        comparison.forEach(result => {
            const statusClass = result.status === 'success' ? 'success' : 'danger';
            const confidencePercent = (result.confidence * 100).toFixed(1);
            
            html += `
                <tr>
                    <td>
                        <span class="badge bg-${getModelColor(result.model)}">
                            ${formatModelName(result.model)}
                        </span>
                    </td>
                    <td>
                        <span class="badge bg-${getCategoryColor(result.prediction)}">
                            ${result.prediction}
                        </span>
                    </td>
                    <td>
                        <div class="d-flex align-items-center">
                            <div class="progress flex-grow-1 me-2" style="height: 8px;">
                                <div class="progress-bar bg-${statusClass}" 
                                     style="width: ${confidencePercent}%">
                                </div>
                            </div>
                            <span>${confidencePercent}%</span>
                        </div>
                    </td>
                    <td>
                        <span class="badge bg-${statusClass}">
                            ${result.status === 'success' ? '✓ Success' : '✗ Error'}
                        </span>
                    </td>
                    <td>
                        ${result.error ? 
                            `<small class="text-danger">${result.error}</small>` :
                            ''}
                        ${result.model === 'svm' && result.params ? 
                            `<div><small>Kernel: ${result.params.kernel}, C: ${result.params.C}</small></div>` : 
                            ''}
                        ${!result.error && !(result.model === 'svm' && result.params) ? 
                            '<small class="text-muted">No additional info</small>' : 
                            ''}
                    </td>
                </tr>
            `;
        });
        
        html += `
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="mt-3">
                        <h6>Analysis:</h6>
                        <p class="mb-0">
                            ${summary.agreement ? 
                                'All models agree on the prediction.' :
                                'Models disagree on the prediction. Consider reviewing the individual results.'}
                        </p>
                    </div>
                </div>
            </div>
        `;
        
        comparisonContainer.innerHTML = html;
    }
    
    // Create comparison chart
    createComparisonChart() {
        const ctx = document.getElementById('comparisonChart');
        if (!ctx || !this.comparisonData) return;
        
        // Destroy existing chart
        if (this.comparisonChart) {
            this.comparisonChart.destroy();
        }
        
        const { comparison } = this.comparisonData;
        
        // Filter successful predictions
        const successfulComparisons = comparison.filter(c => c.status === 'success');
        
        const modelNames = successfulComparisons.map(c => formatModelName(c.model));
        const confidences = successfulComparisons.map(c => c.confidence * 100);
        const predictions = successfulComparisons.map(c => c.prediction);
        
        // Colors based on model
        const colors = modelNames.map(name => {
            if (name.includes('Logistic')) return '#3b82f6';
            if (name.includes('Naive')) return '#8b5cf6';
            if (name.includes('SVM')) return '#f59e0b';
            if (name.includes('LSTM')) return '#10b981';
            return '#6b7280';
        });
        
        this.comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [{
                    label: 'Confidence (%)',
                    data: confidences,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.2', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const idx = context.dataIndex;
                                return [
                                    `Model: ${modelNames[idx]}`,
                                    `Prediction: ${predictions[idx]}`,
                                    `Confidence: ${confidences[idx].toFixed(1)}%`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Confidence (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Models'
                        }
                    }
                }
            }
        });
    }
    
    // Show comparison modal
    showComparisonModal(smsText) {
        if (!smsText.trim()) {
            showAlert('Please enter an SMS message to compare models.', 'warning');
            return;
        }
        
        // Update modal title
        const modalTitle = document.getElementById('comparisonModalLabel');
        if (modalTitle) {
            modalTitle.textContent = `Model Comparison: "${smsText.substring(0, 50)}${smsText.length > 50 ? '...' : ''}"`;
        }
        
        // Show loading
        const comparisonContainer = document.getElementById('comparisonContainer');
        if (comparisonContainer) {
            comparisonContainer.innerHTML = `
                <div class="text-center py-5">
                    <div class="loading-spinner"></div>
                    <p class="text-muted mt-3">Comparing models...</p>
                </div>
            `;
        }
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('comparisonModal'));
        modal.show();
        
        // Perform comparison
        this.compareModels(smsText).then(() => {
            // Comparison completed
        });
    }
}

// Helper functions
function getModelColor(modelCode) {
    const colors = {
        'lr': 'primary',
        'nb': 'secondary',
        'svm': 'warning',
        'lstm': 'success'
    };
    return colors[modelCode] || 'secondary';
}

function getCategoryColor(category) {
    const cat = category.toLowerCase();
    if (cat.includes('spam') || cat.includes('scam')) return 'danger';
    if (cat.includes('phishing')) return 'warning';
    if (cat.includes('promotional')) return 'info';
    if (cat.includes('legitimate')) return 'success';
    return 'secondary';
}

function formatModelName(modelCode) {
    const names = {
        'lr': 'Logistic Regression',
        'nb': 'Naive Bayes',
        'svm': 'SVM',
        'lstm': 'LSTM'
    };
    return names[modelCode] || modelCode;
}

// Initialize comparison module
const modelComparison = new ModelComparison();

// Export for global use
window.modelComparison = modelComparison;
window.showComparisonModal = (smsText) => modelComparison.showComparisonModal(smsText);