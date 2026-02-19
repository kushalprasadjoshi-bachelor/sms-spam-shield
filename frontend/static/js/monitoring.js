let throughputChart, modelPieChart, categoryChart, confidenceChart, feedbackModelChart;

async function refreshDashboard() {
    try {
        const response = await fetch('/api/v1/dashboard');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();

        // Safely extract data with defaults
        const totalPredictions = data.total_predictions || 0;
        const modelCounts = data.model_counts || {};
        const categoryDist = data.category_distribution || {};
        const avgConf = data.average_confidence || {};
        const throughputData = data.throughput || [];
        const rawAccuracy = data.accuracy;
        const feedbackStats = data.feedback_stats || {};
        const totalCorrections = feedbackStats.total_corrections || 0;
        const totalFeedback = feedbackStats.total_feedback || 0;
        const totalConfirmedCorrect = feedbackStats.total_confirmed_correct || 0;
        const correctionsByModel = feedbackStats.corrections_by_model || {};
        const normalizedAccuracy = (rawAccuracy !== null && rawAccuracy !== undefined && Number.isFinite(Number(rawAccuracy)))
            ? Number(rawAccuracy)
            : null;
        const feedbackAccuracy = totalFeedback > 0
            ? totalConfirmedCorrect / totalFeedback
            : null;
        const accuracy = normalizedAccuracy ?? feedbackAccuracy;

        // Update cards
        const dashboardCards = document.getElementById('dashboard-cards');
        if (dashboardCards) {
            const cardsHtml = `
                <div class="col-md-3">
                    <div class="card text-white bg-primary mb-3">
                        <div class="card-body">
                            <h5 class="card-title">Total Predictions</h5>
                            <p class="card-text display-6">${totalPredictions}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-white bg-success mb-3">
                        <div class="card-body">
                            <h5 class="card-title">Accuracy</h5>
                            <p class="card-text display-6">${accuracy !== null ? (accuracy * 100).toFixed(1) + '%' : 'N/A'}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-white bg-info mb-3">
                        <div class="card-body">
                            <h5 class="card-title">Models Active</h5>
                            <p class="card-text display-6">${Object.keys(modelCounts).length}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-white bg-warning mb-3">
                        <div class="card-body">
                            <h5 class="card-title">Categories</h5>
                            <p class="card-text display-6">${Object.keys(categoryDist).length}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-white bg-danger mb-3">
                        <div class="card-body">
                            <h5 class="card-title">Corrections</h5>
                            <p class="card-text display-6">${totalCorrections}</p>
                            <small>${totalFeedback} feedback submitted</small>
                        </div>
                    </div>
                </div>
            `;
            dashboardCards.innerHTML = cardsHtml;
        }

        // Throughput chart
        const throughputEl = document.getElementById('throughputChart');
        if (throughputEl) {
            const throughputCtx = throughputEl.getContext('2d');
            if (throughputChart) throughputChart.destroy();
            throughputChart = new Chart(throughputCtx, {
                type: 'line',
                data: {
                    labels: throughputData.map(d => new Date(d.time * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })),
                    datasets: [{
                        label: 'Predictions per minute',
                        data: throughputData.map(d => d.count),
                        borderColor: '#3b82f6',
                        tension: 0.1
                    }]
                }
            });
        }

        // Model pie chart
        const modelPieEl = document.getElementById('modelPieChart');
        if (modelPieEl) {
            const modelCtx = modelPieEl.getContext('2d');
            if (modelPieChart) modelPieChart.destroy();
            modelPieChart = new Chart(modelCtx, {
                type: 'pie',
                data: {
                    labels: Object.keys(modelCounts),
                    datasets: [{
                        data: Object.values(modelCounts),
                        backgroundColor: ['#3b82f6', '#8b5cf6', '#f59e0b', '#10b981']
                    }]
                }
            });
        }

        // Category bar chart
        const categoryEl = document.getElementById('categoryChart');
        if (categoryEl) {
            const catCtx = categoryEl.getContext('2d');
            if (categoryChart) categoryChart.destroy();
            categoryChart = new Chart(catCtx, {
                type: 'bar',
                data: {
                    labels: Object.keys(categoryDist),
                    datasets: [{
                        label: 'Count',
                        data: Object.values(categoryDist),
                        backgroundColor: '#f59e0b'
                    }]
                }
            });
        }

        // Confidence bar chart
        const confidenceEl = document.getElementById('confidenceChart');
        if (confidenceEl) {
            const confCtx = confidenceEl.getContext('2d');
            if (confidenceChart) confidenceChart.destroy();
            confidenceChart = new Chart(confCtx, {
                type: 'bar',
                data: {
                    labels: Object.keys(avgConf),
                    datasets: [{
                        label: 'Avg Confidence',
                        data: Object.values(avgConf).map(v => v * 100),
                        backgroundColor: '#10b981'
                    }]
                },
                options: {
                    scales: { y: { beginAtZero: true, max: 100 } }
                }
            });
        }

        // Feedback corrections by model
        const feedbackEl = document.getElementById('feedbackModelChart');
        if (feedbackEl) {
            const feedbackCtx = feedbackEl.getContext('2d');
            if (feedbackModelChart) feedbackModelChart.destroy();

            const labels = Object.keys(correctionsByModel);
            const values = Object.values(correctionsByModel);

            feedbackModelChart = new Chart(feedbackCtx, {
                type: 'bar',
                data: {
                    labels: labels.length > 0 ? labels : ['No corrections yet'],
                    datasets: [{
                        label: 'Corrections',
                        data: values.length > 0 ? values : [0],
                        backgroundColor: '#ef4444'
                    }]
                },
                options: {
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
        }

    } catch (error) {
        console.error('Failed to load dashboard:', error);
    }
}

// Only run on dashboard page (if dashboard-cards element exists)
if (document.getElementById('dashboard-cards')) {
    setInterval(refreshDashboard, 5000);
    document.addEventListener('DOMContentLoaded', refreshDashboard);
}
