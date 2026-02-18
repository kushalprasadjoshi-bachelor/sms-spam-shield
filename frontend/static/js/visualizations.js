// Visualizations and model theory for SMS Spam Shield

document.addEventListener('DOMContentLoaded', () => {
    initializeMetricsChart();
    updateModelTheoryFromSelection();
});

function initializeMetricsChart() {
    const canvas = document.getElementById('metricsChart');
    if (!canvas) return;

    window.metricsChart = new Chart(canvas, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [{
                label: 'Model Performance',
                data: [0, 0, 0, 0],
                backgroundColor: 'rgba(59, 130, 246, 0.2)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(59, 130, 246, 1)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 6
            }]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        callback(value) {
                            return `${value}%`;
                        }
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label(context) {
                            return `${context.dataset.label}: ${context.parsed.r.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

function updateModelTheoryFromSelection() {
    const selected = AppState?.activeModel || [...(AppState?.selectedModels || [])][0] || 'lr';
    loadModelTheory(selected);
}

function loadModelTheory(modelId) {
    const theoryContent = document.getElementById('modelTheory');
    const mathContent = document.getElementById('modelMath');
    if (!theoryContent || !mathContent) return;

    theoryContent.innerHTML = getModelTheory(modelId);
    mathContent.innerHTML = getModelMathematics(modelId);
    updateMetricsChartForModel(modelId);
}

function getModelTheory(modelId) {
    const theoryMap = {
        lr: `
            <h6>Logistic Regression</h6>
            <p>Linear classifier that estimates class probabilities through a logistic link function.</p>
            <ul>
                <li>Fast and interpretable baseline for text vectors.</li>
                <li>Works well with TF-IDF features.</li>
                <li>Regularization helps prevent overfitting.</li>
            </ul>
        `,
        nb: `
            <h6>Naive Bayes</h6>
            <p>Probabilistic classifier based on Bayes' rule and conditional independence assumptions.</p>
            <ul>
                <li>Strong baseline for short-message text classification.</li>
                <li>Very fast training and inference.</li>
                <li>Assumption of independent features can reduce accuracy on correlated tokens.</li>
            </ul>
        `,
        svm: `
            <h6>Support Vector Machine</h6>
            <p>Margin-based classifier that finds a separating hyperplane with maximum gap between classes.</p>
            <ul>
                <li>Often strong on sparse high-dimensional text vectors.</li>
                <li>Kernel choice controls non-linearity.</li>
                <li>Probability estimates may require calibration.</li>
            </ul>
        `,
        lstm: `
            <h6>LSTM (Long Short-Term Memory)</h6>
            <p>Sequence model that keeps context through gated memory cells.</p>
            <ul>
                <li>Captures token order and long dependencies.</li>
                <li>Supports token-level explanation from attention signals.</li>
                <li>Higher compute and tighter version compatibility requirements.</li>
            </ul>
        `
    };
    return theoryMap[modelId] || '<p>Theory is not available for this model.</p>';
}

function getModelMathematics(modelId) {
    const mathMap = {
        lr: `
            <h6>Probability Model</h6>
            <div class="math-equation">
                p(y=1|x) = 1 / (1 + e<sup>-(w<sup>T</sup>x + b)</sup>)
            </div>
            <h6 class="mt-3">Loss</h6>
            <div class="math-equation">
                J = - (1/m) &sum; [ y log(p) + (1-y) log(1-p) ]
            </div>
        `,
        nb: `
            <h6>Bayes Rule</h6>
            <div class="math-equation">
                P(y|x) = P(x|y) P(y) / P(x)
            </div>
            <h6 class="mt-3">Naive Factorization</h6>
            <div class="math-equation">
                P(x|y) = &prod;<sub>i</sub> P(x<sub>i</sub>|y)
            </div>
        `,
        svm: `
            <h6>Soft-Margin Objective</h6>
            <div class="math-equation">
                minimize 1/2 ||w||<sup>2</sup> + C &sum; &xi;<sub>i</sub>
            </div>
            <h6 class="mt-3">Constraints</h6>
            <div class="math-equation">
                y<sub>i</sub>(w<sup>T</sup>x<sub>i</sub> + b) &ge; 1 - &xi;<sub>i</sub>, &xi;<sub>i</sub> &ge; 0
            </div>
        `,
        lstm: `
            <h6>Gate Equations</h6>
            <div class="math-equation">
                f<sub>t</sub> = &sigma;(W<sub>f</sub>[h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>f</sub>)<br>
                i<sub>t</sub> = &sigma;(W<sub>i</sub>[h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>i</sub>)<br>
                o<sub>t</sub> = &sigma;(W<sub>o</sub>[h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>o</sub>)
            </div>
            <h6 class="mt-3">Cell Update</h6>
            <div class="math-equation">
                C<sub>t</sub> = f<sub>t</sub> * C<sub>t-1</sub> + i<sub>t</sub> * C&#771;<sub>t</sub><br>
                h<sub>t</sub> = o<sub>t</sub> * tanh(C<sub>t</sub>)
            </div>
        `
    };
    return mathMap[modelId] || '<p>Mathematics is not available for this model.</p>';
}

function updateMetricsChartForModel(modelId) {
    if (!window.metricsChart) return;

    const modelMetrics = AppState?.modelInfo?.[modelId];
    const values = [
        Number(modelMetrics?.accuracy || 0),
        Number(modelMetrics?.precision || 0),
        Number(modelMetrics?.recall || 0),
        Number(modelMetrics?.f1_score || 0)
    ].map(value => Math.max(0, Math.min(100, value * 100)));

    window.metricsChart.data.datasets[0].label = `${getModelLabel(modelId)} Metrics`;
    window.metricsChart.data.datasets[0].data = values;
    window.metricsChart.update();
}

function getModelLabel(modelId) {
    const labels = {
        lr: 'Logistic Regression',
        nb: 'Naive Bayes',
        svm: 'SVM',
        lstm: 'LSTM'
    };
    return labels[modelId] || modelId;
}

function createConfidenceAnimation(elementId, targetConfidence, duration = 1000) {
    const element = document.getElementById(elementId);
    if (!element) return;

    let currentConfidence = 0;
    const startTime = Date.now();

    function animate() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeOut = 1 - Math.pow(1 - progress, 3);
        currentConfidence = targetConfidence * easeOut;
        element.style.width = `${currentConfidence * 100}%`;
        if (progress < 1) requestAnimationFrame(animate);
    }

    animate();
}

function createWordCloud(words, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    const sortedWords = [...words].sort((a, b) => b.importance - a.importance);
    sortedWords.forEach(wordData => {
        const span = document.createElement('span');
        span.className = 'word-cloud-word';
        span.textContent = wordData.word;
        const fontSize = 12 + (wordData.importance * 36);
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
        span.onmouseenter = () => { span.style.transform = 'scale(1.1)'; };
        span.onmouseleave = () => { span.style.transform = 'scale(1)'; };
        container.appendChild(span);
    });
}

window.loadModelTheory = loadModelTheory;
window.updateModelTheoryFromSelection = updateModelTheoryFromSelection;
window.createConfidenceAnimation = createConfidenceAnimation;
window.createWordCloud = createWordCloud;
