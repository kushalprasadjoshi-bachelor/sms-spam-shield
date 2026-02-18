// Model comparison functionality for SMS Spam Shield

class ModelComparison {
    constructor() {
        this.comparisonData = null;
        this.comparisonChart = null;
        this.currentSms = '';
        this.selectedMethod = 'weighted_voting';
        this.ensembleSummary = null;
        this.bindEvents();
    }

    bindEvents() {
        document.addEventListener('DOMContentLoaded', () => {
            const methodSelect = document.getElementById('comparisonEnsembleMethod');
            if (methodSelect) {
                methodSelect.addEventListener('change', event => {
                    this.selectedMethod = event.target.value;
                    this.updateEnsembleSummary();
                });
            }
        });
    }

    async compareModels(smsText, selectedModels) {
        try {
            const response = await APIService.compareModels(smsText, selectedModels);
            this.currentSms = smsText;
            this.comparisonData = response;
            this.updateComparisonUI();
            this.createComparisonChart();
            this.updateEnsembleSummary();
            return response;
        } catch (error) {
            console.error('Model comparison failed:', error);
            showAlert(`Comparison failed: ${error.message}`, 'danger');
            return null;
        }
    }

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
                                    ${summary.agreement ? '<i class="fas fa-check text-success"></i>' : '<i class="fas fa-times text-danger"></i>'}
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
                                <div class="progress-bar bg-${statusClass}" style="width: ${confidencePercent}%"></div>
                            </div>
                            <span>${confidencePercent}%</span>
                        </div>
                    </td>
                    <td>
                        <span class="badge bg-${statusClass}">
                            ${result.status === 'success' ? '&#10003; Success' : '&#10007; Error'}
                        </span>
                    </td>
                    <td>
                        ${result.error ? `<small class="text-danger">${result.error}</small>` : ''}
                        ${result.model === 'svm' && result.params ? `<div><small>Kernel: ${result.params.kernel}, C: ${result.params.C}</small></div>` : ''}
                        ${!result.error && !(result.model === 'svm' && result.params) ? '<small class="text-muted">No additional info</small>' : ''}
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
                            ${summary.agreement ? 'All models agree on the prediction.' : 'Models disagree. Review confidence and selected ensemble method.'}
                        </p>
                    </div>
                </div>
            </div>
        `;

        comparisonContainer.innerHTML = html;
    }

    createComparisonChart() {
        const canvas = document.getElementById('comparisonChartModal');
        if (!canvas || !this.comparisonData) return;

        if (this.comparisonChart) {
            this.comparisonChart.destroy();
        }

        const successfulComparisons = this.comparisonData.comparison.filter(item => item.status === 'success');
        if (successfulComparisons.length === 0) return;

        const modelNames = successfulComparisons.map(item => formatModelName(item.model));
        const confidences = successfulComparisons.map(item => item.confidence * 100);
        const predictions = successfulComparisons.map(item => item.prediction);
        const colors = modelNames.map(name => {
            if (name.includes('Logistic')) return '#3b82f6';
            if (name.includes('Naive')) return '#8b5cf6';
            if (name.includes('SVM')) return '#f59e0b';
            if (name.includes('LSTM')) return '#10b981';
            return '#6b7280';
        });

        this.comparisonChart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [{
                    label: 'Confidence (%)',
                    data: confidences,
                    backgroundColor: colors,
                    borderColor: colors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label(context) {
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
                        title: { display: true, text: 'Confidence (%)' }
                    },
                    x: {
                        title: { display: true, text: 'Models' }
                    }
                }
            }
        });
    }

    calculateWeightedVoting(successfulComparisons) {
        const scores = {};
        successfulComparisons.forEach(item => {
            const label = item.prediction;
            scores[label] = (scores[label] || 0) + item.confidence;
        });
        const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
        const total = Object.values(scores).reduce((a, b) => a + b, 0) || 1;
        return {
            method: 'weighted_voting',
            prediction: best[0],
            confidence: best[1] / total
        };
    }

    calculateAveraging(successfulComparisons) {
        const grouped = {};
        successfulComparisons.forEach(item => {
            const label = item.prediction;
            if (!grouped[label]) grouped[label] = [];
            grouped[label].push(item.confidence);
        });
        const averages = Object.entries(grouped).map(([label, values]) => ({
            label,
            avg: values.reduce((a, b) => a + b, 0) / values.length
        }));
        averages.sort((a, b) => b.avg - a.avg);
        return {
            method: 'averaging',
            prediction: averages[0].label,
            confidence: averages[0].avg
        };
    }

    updateEnsembleSummary() {
        const summaryElement = document.getElementById('comparisonEnsembleSummary');
        if (!summaryElement) return;

        if (!this.comparisonData) {
            summaryElement.textContent = 'Run a comparison first.';
            return;
        }

        const successfulComparisons = this.comparisonData.comparison.filter(item => item.status === 'success');
        if (successfulComparisons.length === 0) {
            summaryElement.textContent = 'No successful model outputs to calculate ensemble result.';
            return;
        }

        const result = this.selectedMethod === 'averaging'
            ? this.calculateAveraging(successfulComparisons)
            : this.calculateWeightedVoting(successfulComparisons);

        this.ensembleSummary = result;
        summaryElement.innerHTML = `
            Ensemble prediction: <strong>${result.prediction}</strong><br>
            Confidence: <strong>${(result.confidence * 100).toFixed(1)}%</strong>
        `;
    }

    showComparisonModal(smsText) {
        if (!smsText.trim()) {
            showAlert('Please enter an SMS message to compare models.', 'warning');
            return;
        }

        const selectedModels = [...(AppState?.selectedModels || [])];
        if (selectedModels.length < 2) {
            showAlert('Select at least two models to compare.', 'warning');
            return;
        }

        const modalTitle = document.getElementById('comparisonModalLabel');
        if (modalTitle) {
            const safePreview = smsText.substring(0, 50);
            modalTitle.textContent = `Model Comparison (${selectedModels.length} selected): "${safePreview}${smsText.length > 50 ? '...' : ''}"`;
        }

        const comparisonContainer = document.getElementById('comparisonContainer');
        if (comparisonContainer) {
            comparisonContainer.innerHTML = `
                <div class="text-center py-5">
                    <div class="loading-spinner"></div>
                    <p class="text-muted mt-3">Comparing models...</p>
                </div>
            `;
        }

        const modal = new bootstrap.Modal(document.getElementById('comparisonModal'));
        modal.show();
        this.compareModels(smsText, selectedModels);
    }

    exportComparisonData() {
        if (!this.comparisonData) {
            showAlert('No comparison data available to export.', 'warning');
            return;
        }

        const payload = {
            exported_at: new Date().toISOString(),
            sms: this.currentSms,
            method: this.selectedMethod,
            ensemble_summary: this.ensembleSummary,
            comparison_data: this.comparisonData
        };

        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `comparison_${Date.now()}.json`;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
    }
}

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
    const cat = String(category || '').toLowerCase();
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

const modelComparison = new ModelComparison();

window.modelComparison = modelComparison;
window.showComparisonModal = smsText => modelComparison.showComparisonModal(smsText);
window.exportComparisonData = () => modelComparison.exportComparisonData();
