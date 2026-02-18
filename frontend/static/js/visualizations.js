// Visualizations for SMS Spam Shield

// Initialize visualizations when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeMetricsChart();
    loadModelTheory('lr'); // Load Logistic Regression by default
});

// Initialize metrics chart
function initializeMetricsChart() {
    const ctx = document.getElementById('metricsChart');
    if (!ctx) return;
    
    // Sample data - this would be populated from actual model metrics
    const sampleData = {
        labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        datasets: [{
            label: 'Model Performance',
            data: [85, 82, 88, 85],
            backgroundColor: 'rgba(59, 130, 246, 0.2)',
            borderColor: 'rgba(59, 130, 246, 1)',
            borderWidth: 2,
            pointBackgroundColor: 'rgba(59, 130, 246, 1)',
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            pointRadius: 6
        }]
    };
    
    window.metricsChart = new Chart(ctx, {
        type: 'radar',
        data: sampleData,
        options: {
            responsive: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        callback: function(value) {
                            return value + '%';
                        }
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
                            return `${context.dataset.label}: ${context.parsed.r}%`;
                        }
                    }
                }
            }
        }
    });
}

// Load model theory content
function loadModelTheory(modelId) {
    const theoryContent = document.getElementById('modelTheory');
    const mathContent = document.getElementById('modelMath');
    
    if (!theoryContent || !mathContent) return;
    
    const theory = getModelTheory(modelId);
    const math = getModelMathematics(modelId);
    
    theoryContent.innerHTML = theory;
    mathContent.innerHTML = math;
    
    // Update chart with model-specific data if available
    updateMetricsChartForModel(modelId);
}

// Get model theory HTML
function getModelTheory(modelId) {
    const theories = {
        'lr': `
            <h6>Logistic Regression</h6>
            <p>Logistic Regression is a statistical model that uses a logistic function to model binary or multi-class dependent variables.</p>
            
            <h6 class="mt-3">Key Concepts:</h6>
            <ul>
                <li><strong>Sigmoid Function:</strong> Maps any real-valued number into the range (0, 1)</li>
                <li><strong>Decision Boundary:</strong> Linear boundary separating classes</li>
                <li><strong>Maximum Likelihood Estimation:</strong> Optimizes parameters to maximize likelihood of observed data</li>
            </ul>
            
            <h6 class="mt-3">Advantages:</h6>
            <ul>
                <li>Simple and interpretable</li>
                <li>Provides probability estimates</li>
                <li>Efficient for linearly separable data</li>
                <li>Less prone to overfitting with regularization</li>
            </ul>
            
            <h6 class="mt-3">Limitations:</h6>
            <ul>
                <li>Assumes linear relationship between features and log-odds</li>
                <li>May underperform with complex, non-linear patterns</li>
                <li>Sensitive to outliers</li>
            </ul>
        `,
        
        'nb': `
            <h6>Naive Bayes</h6>
            <p>Naive Bayes classifiers are based on Bayes' theorem with strong independence assumptions between features.</p>
            
            <h6 class="mt-3">Key Concepts:</h6>
            <ul>
                <li><strong>Bayes' Theorem:</strong> P(A|B) = P(B|A) * P(A) / P(B)</li>
                <li><strong>Conditional Independence:</strong> Assumes features are independent given class</li>
                <li><strong>Maximum A Posteriori (MAP):</strong> Selects class with highest posterior probability</li>
            </ul>
            
            <h6 class="mt-3">Advantages:</h6>
            <ul>
                <li>Fast training and prediction</li>
                <li>Works well with high-dimensional data</li>
                <li>Requires less training data</li>
                <li>Good for text classification</li>
            </ul>
            
            <h6 class="mt-3">Limitations:</h6>
            <ul>
                <li>Strong independence assumption may not hold</li>
                <li>Zero-frequency problem (requires smoothing)</li>
                <li>Probability estimates can be inaccurate</li>
            </ul>
        `,
        
        'svm': `
            <h6>Support Vector Machine (SVM)</h6>
            <p>SVM finds the optimal hyperplane that maximizes the margin between classes in high-dimensional space.</p>
            
            <h6 class="mt-3">Key Concepts:</h6>
            <ul>
                <li><strong>Maximum Margin:</strong> Finds hyperplane with greatest separation between classes</li>
                <li><strong>Support Vectors:</strong> Data points closest to the decision boundary</li>
                <li><strong>Kernel Trick:</strong> Maps data to higher dimensions without explicit computation</li>
            </ul>
            
            <h6 class="mt-3">Advantages:</h6>
            <ul>
                <li>Effective in high-dimensional spaces</li>
                <li>Memory efficient (uses only support vectors)</li>
                <li>Versatile with different kernel functions</li>
                <li>Robust against overfitting in high dimensions</li>
            </ul>
            
            <h6 class="mt-3">Limitations:</h6>
            <ul>
                <li>Doesn't perform well with large datasets</li>
                <li>Sensitive to kernel choice and parameters</li>
                <li>Probability estimates require additional computation</li>
                <li>Interpretability decreases with kernel functions</li>
            </ul>
        `,
        
        'lstm': `
            <h6>Long Short-Term Memory (LSTM)</h6>
            <p>LSTM is a type of recurrent neural network capable of learning long-term dependencies in sequence data.</p>
            
            <h6 class="mt-3">Key Concepts:</h6>
            <ul>
                <li><strong>Memory Cells:</strong> Maintain information over long sequences</li>
                <li><strong>Gates:</strong> Control information flow (input, forget, output gates)</li>
                <li><strong>Vanishing Gradient Solution:</strong> Addresses RNN training difficulties</li>
            </ul>
            
            <h6 class="mt-3">Advantages:</h6>
            <ul>
                <li>Captures long-term dependencies</li>
                <li>Effective for sequential data like text</li>
                <li>Can learn complex patterns</li>
                <li>Automatically learns feature representations</li>
            </ul>
            
            <h6 class="mt-3">Limitations:</h6>
            <ul>
                <li>Computationally expensive</li>
                <li>Requires large amounts of training data</li>
                <li>Difficult to interpret (black box)</li>
                <li>Sensitive to hyperparameter tuning</li>
            </ul>
        `
    };
    
    return theories[modelId] || `<p>Theory not available for this model.</p>`;
}

// Get model mathematics HTML
function getModelMathematics(modelId) {
    const mathematics = {
        'lr': `
            <h6>Logistic Function</h6>
            <div class="math-equation">
                P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^{-(wᵀx + b)})
            </div>
            
            <h6 class="mt-3">Cost Function (Cross-Entropy)</h6>
            <div class="math-equation">
                J(w) = -1/m ∑[yⁱ log(h(xⁱ)) + (1-yⁱ) log(1-h(xⁱ))]
            </div>
            
            <h6 class="mt-3">Gradient Update</h6>
            <div class="math-equation">
                wⱼ := wⱼ - α ∂J/∂wⱼ<br>
                where ∂J/∂wⱼ = 1/m ∑(h(xⁱ) - yⁱ)xⱼⁱ
            </div>
            
            <h6 class="mt-3">Multi-class (Softmax)</h6>
            <div class="math-equation">
                P(y=k|x) = e^{wₖᵀx} / ∑ e^{wⱼᵀx}
            </div>
        `,
        
        'nb': `
            <h6>Bayes' Theorem</h6>
            <div class="math-equation">
                P(y|X) = P(X|y) P(y) / P(X)
            </div>
            
            <h6 class="mt-3">Naive Assumption</h6>
            <div class="math-equation">
                P(x₁,x₂,...,xₙ|y) = ∏ P(xᵢ|y)
            </div>
            
            <h6 class="mt-3">Classification Rule</h6>
            <div class="math-equation">
                ŷ = argmax_y P(y) ∏ P(xᵢ|y)
            </div>
            
            <h6 class="mt-3">Gaussian Naive Bayes</h6>
            <div class="math-equation">
                P(xᵢ|y) = 1/√(2πσ_y²) exp(-(xᵢ-μ_y)²/(2σ_y²))
            </div>
        `,
        
        'svm': `
            <h6>Primal Form</h6>
            <div class="math-equation">
                min_{w,b} ½||w||²<br>
                s.t. yⁱ(wᵀxⁱ + b) ≥ 1
            </div>
            
            <h6 class="mt-3">Dual Form</h6>
            <div class="math-equation">
                max_α ∑αᵢ - ½∑∑αᵢαⱼyⁱyⱼxⁱᵀxⱼ<br>
                s.t. αᵢ ≥ 0, ∑αᵢyⁱ = 0
            </div>
            
            <h6 class="mt-3">Kernel Function</h6>
            <div class="math-equation">
                K(xⁱ, xⱼ) = φ(xⁱ)ᵀφ(xⱼ)<br>
                Common kernels: Linear, Polynomial, RBF
            </div>
            
            <h6 class="mt-3">Decision Function</h6>
            <div class="math-equation">
                f(x) = sign(∑αᵢyⁱK(xⁱ, x) + b)
            </div>
        `,
        
        'lstm': `
            <h6>LSTM Cell Equations</h6>
            
            <h6 class="mt-2">Forget Gate</h6>
            <div class="math-equation">
                fₜ = σ(W_f·[h_{t-1}, xₜ] + b_f)
            </div>
            
            <h6 class="mt-2">Input Gate</h6>
            <div class="math-equation">
                iₜ = σ(W_i·[h_{t-1}, xₜ] + b_i)<br>
                Ĉₜ = tanh(W_c·[h_{t-1}, xₜ] + b_c)
            </div>
            
            <h6 class="mt-2">Cell State Update</h6>
            <div class="math-equation">
                Cₜ = fₜ * C_{t-1} + iₜ * Ĉₜ
            </div>
            
            <h6 class="mt-2">Output Gate</h6>
            <div class="math-equation">
                oₜ = σ(W_o·[h_{t-1}, xₜ] + b_o)<br>
                hₜ = oₜ * tanh(Cₜ)
            </div>
        `
    };
    
    return mathematics[modelId] || `<p>Mathematical details not available for this model.</p>`;
}

// SHAP Waterfall Visualization
function createShapWaterfall(shapValues, featureNames, baseValue, prediction, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Sort features by absolute SHAP value
    const features = shapValues.map((value, index) => ({
        name: featureNames[index],
        value: value,
        absValue: Math.abs(value)
    })).sort((a, b) => b.absValue - a.absValue).slice(0, 10);
    
    let html = `
        <div class="shap-waterfall">
            <div class="shap-base-value">
                Base Value: ${baseValue.toFixed(4)}
            </div>
            <div class="shap-features">
    `;
    
    let cumulative = baseValue;
    features.forEach((feature, i) => {
        cumulative += feature.value;
        const color = feature.value > 0 ? 'text-danger' : 'text-success';
        const arrow = feature.value > 0 ? '↑' : '↓';
        
        html += `
            <div class="shap-feature-row d-flex align-items-center mb-2">
                <div class="shap-feature-name" style="width: 150px;">
                    ${feature.name}
                </div>
                <div class="shap-feature-bar flex-grow-1 mx-3">
                    <div class="progress" style="height: 24px;">
                        <div class="progress-bar ${feature.value > 0 ? 'bg-danger' : 'bg-success'}" 
                             style="width: ${Math.abs(feature.value) * 50}%;">
                            ${feature.value > 0 ? '+' : ''}${feature.value.toFixed(4)}
                        </div>
                    </div>
                </div>
                <div class="shap-cumulative ${color} fw-bold">
                    ${arrow} ${cumulative.toFixed(4)}
                </div>
            </div>
        `;
    });
    
    html += `
            </div>
            <div class="shap-prediction mt-3">
                <strong>Final Prediction:</strong> ${prediction}
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// Add SVM model card HTML generation
function createSVMModelCard(metrics) {
    return `
        <div class="model-card" id="modelCard-svm">
            <div class="model-card-header">
                <div class="model-icon svm">
                    <i class="fas fa-project-diagram"></i>
                </div>
                <div>
                    <h6 class="mb-0">Support Vector Machine</h6>
                    <small class="text-muted">${metrics.kernel || 'RBF'} Kernel</small>
                </div>
                <div class="ms-auto">
                    <span class="badge bg-warning">Hyperparameter Tuned</span>
                </div>
            </div>
            <div class="model-metrics">
                <div class="metric">
                    <div class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${(metrics.f1_score * 100).toFixed(1)}%</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
            <div class="mt-2 small">
                <i class="fas fa-tachometer-alt"></i> CV Score: ${(metrics.best_cv_score * 100).toFixed(1)}%
            </div>
            <div class="mt-1">
                <button class="btn btn-sm btn-outline-warning" onclick="showSVMDetails()">
                    <i class="fas fa-cog"></i> View Params
                </button>
            </div>
        </div>
    `;
}

// Update metrics chart for specific model
function updateMetricsChartForModel(modelId) {
    if (!window.metricsChart) return;
    
    // Sample data for different models
    const modelMetrics = {
        'lr': [92, 91, 93, 92],
        'nb': [89, 87, 90, 88],
        'svm': [94, 93, 95, 94],
        'lstm': [96, 95, 97, 96]
    };
    
    const metrics = modelMetrics[modelId] || [85, 82, 88, 85];
    
    window.metricsChart.data.datasets[0].data = metrics;
    window.metricsChart.update();
}

// Create real-time confidence animation
function createConfidenceAnimation(elementId, targetConfidence, duration = 1000) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    let currentConfidence = 0;
    const startTime = Date.now();
    
    function animate() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth animation
        const easeOut = 1 - Math.pow(1 - progress, 3);
        currentConfidence = targetConfidence * easeOut;
        
        element.style.width = `${currentConfidence * 100}%`;
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }
    
    animate();
}

// Create word cloud visualization
function createWordCloud(words, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = '';
    
    // Sort words by importance
    const sortedWords = [...words].sort((a, b) => b.importance - a.importance);
    
    // Create word elements
    sortedWords.forEach(wordData => {
        const span = document.createElement('span');
        span.className = 'word-cloud-word';
        span.textContent = wordData.word;
        
        // Size based on importance
        const fontSize = 12 + (wordData.importance * 36); // 12px to 48px
        const opacity = 0.5 + (wordData.importance * 0.5);
        
        span.style.fontSize = `${fontSize}px`;
        span.style.opacity = opacity;
        span.style.padding = '4px 8px';
        span.style.margin = '2px';
        span.style.display = 'inline-block';
        span.style.color = `rgba(239, 68, 68, ${opacity})`;
        span.style.fontWeight = 'bold';
        span.style.cursor = 'pointer';
        span.style.transition = 'transform 0.2s';
        
        span.title = `Importance: ${(wordData.importance * 100).toFixed(1)}%`;
        
        span.onmouseenter = () => {
            span.style.transform = 'scale(1.1)';
        };
        
        span.onmouseleave = () => {
            span.style.transform = 'scale(1)';
        };
        
        container.appendChild(span);
    });
}

// Export visualization functions
window.loadModelTheory = loadModelTheory;
window.createConfidenceAnimation = createConfidenceAnimation;
window.createWordCloud = createWordCloud;