// Main application JavaScript for SMS Spam Shield

const AVAILABLE_MODELS = ['lr', 'nb', 'svm', 'lstm'];
const sidebarState = {
    leftCollapsed: localStorage.getItem('leftSidebarCollapsed') === 'true',
    rightCollapsed: localStorage.getItem('rightSidebarCollapsed') === 'true'
};

// Application state
const AppState = {
    selectedModels: new Set(['lr']),
    chatHistory: [],
    modelInfo: {},
    theme: 'light',
    historyFilter: 'all',
    activeModel: 'lr',

    init() {
        const savedModels = localStorage.getItem('selectedModels');
        if (savedModels) {
            this.selectedModels = new Set(JSON.parse(savedModels));
        }

        const savedHistory = localStorage.getItem('chatHistory');
        if (savedHistory) {
            this.chatHistory = JSON.parse(savedHistory);
        }

        this.theme = localStorage.getItem('theme') || 'light';
        this.historyFilter = localStorage.getItem('historyFilter') || 'all';
        this.activeModel = localStorage.getItem('activeModel') || [...this.selectedModels][0] || 'lr';
        this.applyTheme();
    },

    save() {
        localStorage.setItem('selectedModels', JSON.stringify([...this.selectedModels]));
        localStorage.setItem('chatHistory', JSON.stringify(this.chatHistory));
        localStorage.setItem('theme', this.theme);
        localStorage.setItem('historyFilter', this.historyFilter);
        localStorage.setItem('activeModel', this.activeModel);
    },

    toggleTheme() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        this.applyTheme();
        this.save();
    },

    applyTheme() {
        const toggle = document.getElementById('darkModeToggle');
        if (this.theme === 'dark') {
            document.documentElement.setAttribute('data-bs-theme', 'dark');
            if (toggle) toggle.checked = true;
        } else {
            document.documentElement.removeAttribute('data-bs-theme');
            if (toggle) toggle.checked = false;
        }
    },

    addMessage(message) {
        this.chatHistory.push(message);
        if (this.chatHistory.length > 50) {
            this.chatHistory = this.chatHistory.slice(-50);
        }
        this.save();
    },

    getStats() {
        const assistantMessages = this.chatHistory.filter(msg => msg.type === 'assistant' && msg.result);
        const total = assistantMessages.length;
        const spamCount = assistantMessages.filter(msg => {
            const category = msg.result.category || '';
            return ['spam', 'phishing', 'scam'].some(tag => category.toLowerCase().includes(tag));
        }).length;

        const confidences = assistantMessages
            .map(msg => Number(msg.result.confidence))
            .filter(value => !Number.isNaN(value));

        const avgConfidence = confidences.length > 0
            ? confidences.reduce((a, b) => a + b, 0) / confidences.length
            : 0;

        return {
            total,
            spamCount,
            avgConfidence
        };
    }
};

function getFirstSelectedModel() {
    return [...AppState.selectedModels][0] || 'lr';
}

function getModelDisplayName(modelId) {
    const names = {
        lr: 'Logistic Regression',
        nb: 'Naive Bayes',
        svm: 'SVM',
        lstm: 'LSTM'
    };
    return names[modelId] || modelId;
}

function getAssistantForMessage(messageId) {
    return AppState.chatHistory.find(msg => msg.id === messageId && msg.type === 'assistant');
}

function setShellHeights() {
    const navbar = document.getElementById('topNavbar');
    const footer = document.querySelector('.footer');
    const navbarHeight = navbar ? navbar.offsetHeight : 56;
    const footerHeight = footer ? footer.offsetHeight : 120;

    document.documentElement.style.setProperty('--navbar-height', `${navbarHeight}px`);
    document.documentElement.style.setProperty('--footer-height', `${footerHeight}px`);
}

function applyFooterState() {
    const footer = document.getElementById('appFooter');
    const showBtn = document.getElementById('showFooterBtn');
    if (!footer || !showBtn) return;

    const hidden = localStorage.getItem('footerHidden') === 'true';
    footer.classList.toggle('d-none', hidden);
    showBtn.classList.toggle('d-none', !hidden);
    setShellHeights();
}

function toggleFooter(show = null) {
    const currentHidden = localStorage.getItem('footerHidden') === 'true';
    const nextHidden = show === null ? !currentHidden : !show;
    localStorage.setItem('footerHidden', String(nextHidden));
    applyFooterState();
}

function setNavbarScrolledState() {
    const navbar = document.getElementById('topNavbar');
    if (!navbar) return;
    const leftSidebar = document.getElementById('leftSidebar');
    const rightSidebar = document.getElementById('rightSidebar');
    const chatMessages = document.getElementById('chatMessages');
    const hasInternalScroll =
        (leftSidebar && leftSidebar.scrollTop > 12) ||
        (rightSidebar && rightSidebar.scrollTop > 12) ||
        (chatMessages && chatMessages.scrollTop > 12);

    if (window.scrollY > 12 || hasInternalScroll) {
        navbar.classList.add('navbar-scrolled');
    } else {
        navbar.classList.remove('navbar-scrolled');
    }
}

function applySidebarLayout() {
    const leftSidebar = document.getElementById('leftSidebar');
    const rightSidebar = document.getElementById('rightSidebar');
    const mainContent = document.getElementById('mainContent');
    const leftBtn = document.getElementById('toggleLeftSidebar');
    const rightBtn = document.getElementById('toggleRightSidebar');
    if (!leftSidebar || !rightSidebar || !mainContent) return;

    leftSidebar.classList.toggle('d-none', sidebarState.leftCollapsed);
    rightSidebar.classList.toggle('d-none', sidebarState.rightCollapsed);

    mainContent.classList.remove('col-md-6', 'col-lg-8', 'col-md-9', 'col-lg-10', 'col-12');
    if (sidebarState.leftCollapsed && sidebarState.rightCollapsed) {
        mainContent.classList.add('col-12');
    } else if (sidebarState.leftCollapsed || sidebarState.rightCollapsed) {
        mainContent.classList.add('col-md-9', 'col-lg-10');
    } else {
        mainContent.classList.add('col-md-6', 'col-lg-8');
    }

    if (leftBtn) {
        leftBtn.classList.toggle('btn-primary', !sidebarState.leftCollapsed);
        leftBtn.classList.toggle('btn-outline-secondary', sidebarState.leftCollapsed);
    }
    if (rightBtn) {
        rightBtn.classList.toggle('btn-primary', !sidebarState.rightCollapsed);
        rightBtn.classList.toggle('btn-outline-secondary', sidebarState.rightCollapsed);
    }
}

function toggleSidebar(side) {
    if (side === 'left') {
        sidebarState.leftCollapsed = !sidebarState.leftCollapsed;
        localStorage.setItem('leftSidebarCollapsed', String(sidebarState.leftCollapsed));
    } else if (side === 'right') {
        sidebarState.rightCollapsed = !sidebarState.rightCollapsed;
        localStorage.setItem('rightSidebarCollapsed', String(sidebarState.rightCollapsed));
    }
    applySidebarLayout();
}

function updateUI() {
    updateHistoryDisplay();
    updateStats();
    updateModelSelectionUI();
    if (typeof updateModelTheoryFromSelection === 'function') {
        updateModelTheoryFromSelection();
    }
}

function updateHistoryDisplay() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;

    historyList.innerHTML = '';

    const userMessages = AppState.chatHistory.filter(msg => msg.type === 'user');
    const filtered = userMessages.filter(msg => {
        if (AppState.historyFilter === 'all') return true;
        const assistant = getAssistantForMessage(msg.id);
        const category = (assistant?.result?.category || '').toLowerCase();
        return category.includes(AppState.historyFilter.toLowerCase());
    });

    if (filtered.length === 0) {
        historyList.innerHTML = `
            <div class="text-center text-muted py-3" id="emptyHistory">
                <i class="fas fa-inbox fa-2x mb-2"></i>
                <p>No predictions yet</p>
            </div>
        `;
        return;
    }

    const recentHistory = filtered.slice(-10).reverse();
    recentHistory.forEach(message => historyList.appendChild(createHistoryItem(message)));
}

function createHistoryItem(message) {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.onclick = () => loadMessageDetails(message.id);

    const text = message.text.length > 100
        ? `${message.text.substring(0, 100)}...`
        : message.text;

    const assistantMsg = getAssistantForMessage(message.id);
    const category = assistantMsg?.result?.category || 'Unknown';
    const categoryClass = assistantMsg?.result?.category_color || 'secondary';

    div.innerHTML = `
        <div class="history-text">${escapeHtml(text)}</div>
        <div class="d-flex justify-content-between align-items-center">
            <span class="badge bg-${categoryClass}">${escapeHtml(category)}</span>
            <div class="d-flex align-items-center gap-2">
                <small class="text-muted">${message.time}</small>
                <button class="btn btn-sm btn-link text-danger p-0" title="Delete chat" onclick="deleteChat('${message.id}', event)">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `;

    return div;
}

function updateStats() {
    const stats = AppState.getStats();

    const totalEl = document.getElementById('totalPredictions');
    if (totalEl) totalEl.textContent = stats.total;

    const spamEl = document.getElementById('spamCount');
    if (spamEl) spamEl.textContent = stats.spamCount;

    const avgConfidenceEl = document.getElementById('avgConfidence');
    if (avgConfidenceEl) {
        avgConfidenceEl.textContent = `${(stats.avgConfidence * 100).toFixed(1)}%`;
    }

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

function updateModelSelectionUI() {
    const modelAvailability = {};
    AVAILABLE_MODELS.forEach(model => {
        const modelInfo = AppState.modelInfo?.[model];
        modelAvailability[model] = !modelInfo || modelInfo.status === 'loaded';
    });

    AVAILABLE_MODELS.forEach(model => {
        const isSelected = AppState.selectedModels.has(model);
        const isAvailable = modelAvailability[model];

        const card = document.getElementById(`modelCard-${model}`);
        const cardCheck = document.getElementById(`model-${model}`);
        const inlineCheck = document.getElementById(`model-${model}-check`);

        if (card) {
            card.classList.toggle('active', isSelected);
            card.classList.toggle('opacity-50', !isAvailable);
            card.classList.toggle('disabled-model', !isAvailable);
            if (isSelected) {
                card.setAttribute('aria-pressed', 'true');
            } else {
                card.setAttribute('aria-pressed', 'false');
            }
        }

        if (cardCheck) {
            cardCheck.checked = isSelected;
            cardCheck.disabled = !isAvailable;
        }

        if (inlineCheck) {
            inlineCheck.checked = isSelected;
            inlineCheck.disabled = !isAvailable;
        }
    });

    if (!AppState.selectedModels.has(AppState.activeModel)) {
        AppState.activeModel = getFirstSelectedModel();
    }
}

function toggleModel(modelId) {
    const modelInfo = AppState.modelInfo?.[modelId];
    if (modelInfo && modelInfo.status !== 'loaded') {
        showAlert(`${getModelDisplayName(modelId)} is not loaded right now.`, 'warning');
        return;
    }

    if (AppState.selectedModels.has(modelId)) {
        AppState.selectedModels.delete(modelId);
    } else {
        AppState.selectedModels.add(modelId);
    }

    if (AppState.selectedModels.size === 0) {
        AppState.selectedModels.add(modelId);
        showAlert('Select at least one loaded model.', 'warning');
    }

    AppState.activeModel = modelId;
    AppState.save();
    updateUI();
}

function selectModel(modelId) {
    const modelInfo = AppState.modelInfo?.[modelId];
    if (modelInfo && modelInfo.status !== 'loaded') {
        showAlert(`${getModelDisplayName(modelId)} is not loaded right now.`, 'warning');
        return;
    }

    AppState.selectedModels.clear();
    AppState.selectedModels.add(modelId);
    AppState.activeModel = modelId;
    AppState.save();
    updateUI();
}

function loadMessageDetails(messageId) {
    const message = AppState.chatHistory.find(msg => msg.id === messageId && msg.type === 'user');
    if (!message) return;

    const smsInput = document.getElementById('smsInput');
    if (!smsInput) return;

    smsInput.value = message.text;
    smsInput.style.height = 'auto';
    smsInput.style.height = `${smsInput.scrollHeight}px`;
    smsInput.scrollIntoView({ behavior: 'smooth' });
    smsInput.focus();
}

function setActiveHistoryFilterButton(category) {
    const buttons = document.querySelectorAll('.filter-buttons button');
    buttons.forEach(button => button.classList.remove('active'));
    const activeButton = [...buttons].find(button => button.getAttribute('onclick')?.includes(`'${category}'`));
    if (activeButton) activeButton.classList.add('active');
}

function filterHistory(category) {
    AppState.historyFilter = category;
    AppState.save();
    setActiveHistoryFilterButton(category);
    updateHistoryDisplay();
}

function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;

    const welcomeMessage = chatMessages.querySelector('.message.assistant');
    chatMessages.innerHTML = '';
    if (welcomeMessage) {
        chatMessages.appendChild(welcomeMessage);
    }
}

function clearHistory() {
    if (!confirm('Are you sure you want to clear all history?')) return;
    AppState.chatHistory = [];
    AppState.historyFilter = 'all';
    AppState.save();
    clearChat();
    updateUI();
    setActiveHistoryFilterButton('all');
}

function deleteChat(messageId, event = null) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    AppState.chatHistory = AppState.chatHistory.filter(msg => msg.id !== messageId);
    AppState.save();
    updateUI();

    const threadMessages = document.querySelectorAll(`#chatMessages [data-thread-id="${messageId}"]`);
    threadMessages.forEach(element => element.remove());
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function setupEventListeners() {
    const darkToggle = document.getElementById('darkModeToggle');
    if (darkToggle) {
        darkToggle.addEventListener('change', () => AppState.toggleTheme());
    }

    AVAILABLE_MODELS.forEach(model => {
        const cardCheckbox = document.getElementById(`model-${model}`);
        if (cardCheckbox) {
            cardCheckbox.addEventListener('click', event => event.stopPropagation());
            cardCheckbox.addEventListener('change', () => toggleModel(model));
        }

        const inlineCheckbox = document.getElementById(`model-${model}-check`);
        if (inlineCheckbox) {
            inlineCheckbox.addEventListener('change', () => toggleModel(model));
        }
    });

    const smsInput = document.getElementById('smsInput');
    if (smsInput) {
        smsInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = `${this.scrollHeight}px`;
        });

        smsInput.addEventListener('keydown', event => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                if (typeof predictSMS === 'function') predictSMS();
            }
        });
    }

    const leftSidebar = document.getElementById('leftSidebar');
    const rightSidebar = document.getElementById('rightSidebar');
    const chatMessages = document.getElementById('chatMessages');
    [leftSidebar, rightSidebar, chatMessages].forEach(container => {
        if (container) {
            container.addEventListener('scroll', setNavbarScrolledState, { passive: true });
        }
    });

    const leftToggle = document.getElementById('toggleLeftSidebar');
    if (leftToggle) {
        leftToggle.addEventListener('click', () => toggleSidebar('left'));
    }

    const rightToggle = document.getElementById('toggleRightSidebar');
    if (rightToggle) {
        rightToggle.addEventListener('click', () => toggleSidebar('right'));
    }

    window.addEventListener('resize', setShellHeights);
    window.addEventListener('scroll', setNavbarScrolledState, { passive: true });
}

document.addEventListener('DOMContentLoaded', () => {
    AppState.init();
    setupEventListeners();
    applySidebarLayout();
    applyFooterState();
    setShellHeights();
    setNavbarScrolledState();
    updateUI();
    setActiveHistoryFilterButton(AppState.historyFilter);
});

window.toggleModel = toggleModel;
window.selectModel = selectModel;
window.clearChat = clearChat;
window.clearHistory = clearHistory;
window.deleteChat = deleteChat;
window.filterHistory = filterHistory;
window.loadMessageDetails = loadMessageDetails;
window.updateHistoryDisplay = updateHistoryDisplay;
window.updateModelSelectionUI = updateModelSelectionUI;
window.setupEventListeners = setupEventListeners;
window.toggleFooter = toggleFooter;
