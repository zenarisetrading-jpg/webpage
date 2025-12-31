// Configuration matching optimizer
const CONFIG = {
    HARVEST_MIN_CLICKS: 10,
    HARVEST_MIN_ORDERS: 3,
    HARVEST_MIN_SALES: 150.00,
    NEGATIVE_CLICKS_THRESHOLD: 10,
    NEGATIVE_SPEND_THRESHOLD: 10.00,
    ROAS_TARGET: 2.50
};

// Quiz questions
const questions = [
    {
        id: 'spend',
        question: "What's your monthly ad spend?",
        options: [
            { value: '500-2000', label: '$500-$2K', midpoint: 1250 },
            { value: '2000-5000', label: '$2K-$5K', midpoint: 3500 },
            { value: '5000-10000', label: '$5K-$10K', midpoint: 7500 },
            { value: '10000-25000', label: '$10K-$25K', midpoint: 17500 },
            { value: '25000+', label: '$25K+', midpoint: 35000 }
        ]
    },
    {
        id: 'acos',
        question: "What's your current ACOS?",
        options: [
            { value: 'under-15', label: 'Under 15%', penalty: 0, acos_midpoint: 12 },
            { value: '15-25', label: '15-25%', penalty: -5, acos_midpoint: 20 },
            { value: '25-35', label: '25-35%', penalty: -15, acos_midpoint: 30 },
            { value: '35-50', label: '35-50%', penalty: -25, acos_midpoint: 42.5 },
            { value: 'over-50', label: 'Over 50%', penalty: -35, acos_midpoint: 60 },
            { value: 'unknown', label: "Don't know", penalty: -20, acos_midpoint: 30 }
        ]
    },
    {
        id: 'campaigns',
        question: "How many active campaigns?",
        options: [
            { value: '1-5', label: '1-5', complexity: 1.0 },
            { value: '5-15', label: '5-15', complexity: 1.2 },
            { value: '15-30', label: '15-30', complexity: 1.5 },
            { value: '30-50', label: '30-50', complexity: 1.8 },
            { value: '50+', label: '50+', complexity: 2.0 }
        ]
    },
    {
        id: 'negatives',
        question: "Last negative keyword addition?",
        options: [
            { value: 'this-week', label: 'This week', penalty: 0, waste_factor: 0.05 },
            { value: 'this-month', label: 'This month', penalty: -8, waste_factor: 0.12 },
            { value: '2-3-months', label: '2-3 months ago', penalty: -18, waste_factor: 0.22 },
            { value: '6-months', label: '6+ months', penalty: -28, waste_factor: 0.35 },
            { value: 'never', label: 'Never', penalty: -35, waste_factor: 0.45 }
        ]
    },
    {
        id: 'harvest',
        question: "Do you run harvest campaigns?",
        options: [
            { value: 'active', label: 'Yes, actively', penalty: 0, opportunity_factor: 0.05 },
            { value: 'started', label: 'Started, not maintaining', penalty: -12, opportunity_factor: 0.15 },
            { value: 'no', label: 'No', penalty: -20, opportunity_factor: 0.25 },
            { value: 'what', label: "What's that?", penalty: -25, opportunity_factor: 0.30 }
        ]
    },
    {
        id: 'competitor',
        question: "Check for competitor ASINs?",
        options: [
            { value: 'weekly', label: 'Weekly', penalty: 0, waste_factor: 0.05 },
            { value: 'monthly', label: 'Monthly', penalty: -8, waste_factor: 0.12 },
            { value: 'rarely', label: 'Rarely', penalty: -15, waste_factor: 0.20 },
            { value: 'never', label: 'Never', penalty: -25, waste_factor: 0.30 }
        ]
    }
];

let quizAnswers = {};
let selectedFile = null;

// Initialize quiz
function initQuiz() {
    const container = document.getElementById('questionsContainer');
    questions.forEach((q, idx) => {
        const questionDiv = document.createElement('div');
        questionDiv.className = 'question-block';
        questionDiv.innerHTML = `
            <label class="question-label">
                ${idx + 1}. ${q.question}
            </label>
            <div class="options-grid" id="options-${q.id}">
                ${q.options.map(opt => `
                    <div
                        class="quiz-option"
                        data-question="${q.id}"
                        data-value="${opt.value}"
                        onclick='selectAnswer("${q.id}", ${JSON.stringify(opt).replace(/'/g, "&apos;")})'
                    >
                        ${opt.label}
                    </div>
                `).join('')}
            </div>
        `;
        container.appendChild(questionDiv);
    });
}

function selectAnswer(questionId, option) {
    quizAnswers[questionId] = option;

    // Update UI
    const options = document.querySelectorAll(`[data-question="${questionId}"]`);
    options.forEach(opt => {
        if (opt.dataset.value === option.value) {
            opt.classList.add('selected');
        } else {
            opt.classList.remove('selected');
        }
    });

    // Update progress
    const progress = (Object.keys(quizAnswers).length / questions.length) * 100;
    document.getElementById('progressFill').style.width = progress + '%';
    document.getElementById('progressPercent').textContent = Math.round(progress);

    // Enable button when all answered
    document.getElementById('getScoreBtn').disabled = Object.keys(quizAnswers).length < 6;
}

function calculateQuizScore() {
    let score = 100;
    const spend = quizAnswers.spend?.midpoint || 3500;

    score += quizAnswers.acos?.penalty || 0;
    score += quizAnswers.negatives?.penalty || 0;
    score += quizAnswers.harvest?.penalty || 0;
    score += quizAnswers.competitor?.penalty || 0;

    const competitorWasteFactor = quizAnswers.competitor?.waste_factor || 0.15;
    const competitorWaste = Math.round(spend * competitorWasteFactor * 0.35);

    const negativeWasteFactor = quizAnswers.negatives?.waste_factor || 0.15;
    const zeroConversionWaste = Math.round(spend * negativeWasteFactor * 0.25);

    const harvestFactor = quizAnswers.harvest?.opportunity_factor || 0.15;
    const harvestGain = Math.round(spend * harvestFactor * 0.35);

    const targetAcos = 100 / CONFIG.ROAS_TARGET;
    const currentAcos = quizAnswers.acos?.acos_midpoint || 30;
    const bidWaste = currentAcos > targetAcos ?
        Math.round(spend * (currentAcos - targetAcos) / 100 * 0.5) : 0;

    const negativeGaps = Math.round(spend * negativeWasteFactor * 0.10);

    return {
        score: Math.max(45, Math.min(100, score)),
        total: competitorWaste + zeroConversionWaste + harvestGain + bidWaste + negativeGaps,
        breakdown: [
            { title: 'Competitor ASIN Bleed', amount: competitorWaste, priority: 'high' },
            { title: 'Zero-Conversion Keywords', amount: zeroConversionWaste, priority: 'high' },
            { title: 'Missed Harvests', amount: harvestGain, priority: 'high' },
            { title: 'Bid Inefficiency', amount: bidWaste, priority: 'medium' },
            { title: 'Negative Gaps', amount: negativeGaps, priority: 'medium' }
        ].filter(item => item.amount > 0)
    };
}

function showQuizResults() {
    const results = calculateQuizScore();

    // Hide quiz, show results
    document.getElementById('quizSection').classList.add('hidden');
    document.getElementById('resultsSection').classList.remove('hidden');

    // Update score
    const healthScoreEl = document.getElementById('healthScore');
    healthScoreEl.textContent = results.score;

    // Add color class
    if (results.score >= 80) {
        healthScoreEl.style.color = '#10b981';
    } else if (results.score >= 60) {
        healthScoreEl.style.color = '#f59e0b';
    } else {
        healthScoreEl.style.color = '#ef4444';
    }

    // Update opportunity
    const low = Math.round(results.total * 0.85);
    const high = Math.round(results.total * 1.15);
    document.getElementById('opportunityAmount').textContent = `$${low.toLocaleString()} - $${high.toLocaleString()}`;

    // Update breakdown
    const breakdownHTML = results.breakdown.map(item => `
        <div class="breakdown-item ${item.priority}">
            <div class="breakdown-header">
                <span class="breakdown-title">${item.title}</span>
                <span class="breakdown-amount">~$${item.amount.toLocaleString()}/mo</span>
            </div>
        </div>
    `).join('');
    document.getElementById('breakdownList').innerHTML = breakdownHTML;

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showUploadSection() {
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('uploadSection').classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// File upload handlers
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const uploadPrompt = document.getElementById('uploadPrompt');
const fileSelected = document.getElementById('fileSelected');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const analyzeBtn = document.getElementById('analyzeBtn');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '#0891B2';
});

dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = '';
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFileSelect(e.target.files[0]);
});

function handleFileSelect(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
    uploadPrompt.classList.add('hidden');
    fileSelected.classList.remove('hidden');
    analyzeBtn.disabled = false;
    errorMessage.classList.add('hidden');
}

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    errorMessage.classList.add('hidden');

    // Change this to your actual API URL when backend is deployed
    const API_URL = 'http://localhost:8080'; // or 'https://your-api-domain.com'

    try {
        // Hide upload section, show "analyzing" state
        document.getElementById('uploadSection').style.display = 'none';

        // Show processing message
        const processingDiv = document.createElement('section');
        processingDiv.className = 'upload-section';
        processingDiv.id = 'processingSection';
        processingDiv.innerHTML = `
            <div class="container">
                <div class="upload-container" style="text-align: center;">
                    <div style="width: 64px; height: 64px; border: 4px solid var(--bg-secondary); border-top: 4px solid var(--accent-primary); border-radius: 50%; margin: 0 auto 2rem; animation: spin 1s linear infinite;"></div>
                    <h2 style="font-size: 2rem; margin-bottom: 1rem;">Analyzing Your Data...</h2>
                    <p style="color: var(--text-secondary);">Computing waste patterns, harvest opportunities, and optimization potential</p>
                </div>
            </div>
        `;
        document.querySelector('body').appendChild(processingDiv);

        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_URL}/api/analyze`, {
            method: 'POST',
            mode: 'cors',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Analysis failed' }));
            throw new Error(errorData.error || 'Analysis failed');
        }

        const results = await response.json();

        // Remove processing section
        processingDiv.remove();

        // Show results
        showDetailedResults(results);

    } catch (err) {
        // Remove processing section if exists
        const processingSection = document.getElementById('processingSection');
        if (processingSection) processingSection.remove();

        // Show upload section again
        document.getElementById('uploadSection').style.display = 'block';

        let errorMsg = err.message;

        // Provide helpful error messages
        if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
            errorMsg = `
                <strong>Cannot connect to analysis server</strong><br><br>
                The file analysis feature requires a backend server. You have two options:<br><br>
                <strong>Option 1: Set up the backend locally</strong><br>
                1. Navigate to: <code>/Users/zayaanyousuf/Documents/Amazon PPC/microsite applet - audit site</code><br>
                2. Run: <code>python run_audit.py</code><br>
                3. Server will start on port 8080<br><br>
                <strong>Option 2: Contact us for hosted analysis</strong><br>
                Email your report to support@saddle.io for a free manual audit.
            `;
        }

        errorText.innerHTML = errorMsg;
        errorMessage.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});

// Function to show detailed results from backend
function showDetailedResults(data) {
    // Hide all other sections
    document.getElementById('quizSection').classList.add('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('uploadSection').style.display = 'none';

    // Create detailed results section
    const resultsHTML = `
        <section class="results-section">
            <div class="container">
                <div class="results-grid">
                    <!-- Health Score -->
                    <div class="result-card score-card">
                        <h2>Your Actual PPC Health Score</h2>
                        <div class="health-score" style="color: ${getScoreColor(data.healthScore)};">
                            ${data.healthScore}
                        </div>
                        <p class="score-subtitle">Based on ${data.dataQuality.validRows.toLocaleString()} search terms analyzed</p>
                    </div>

                    <!-- Total Opportunity -->
                    <div class="result-card opportunity-card">
                        <div class="card-header">
                            <h3>Total Monthly Opportunity</h3>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="opportunity-icon">
                                <line x1="12" y1="1" x2="12" y2="23"></line>
                                <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
                            </svg>
                        </div>
                        <div class="opportunity-amount">$${data.totalOpportunity.toLocaleString()}</div>
                        <p class="opportunity-subtitle">in recoverable waste + missed gains</p>
                    </div>

                    <!-- Account Overview -->
                    <div class="result-card">
                        <h3 style="margin-bottom: 1.5rem;">Account Overview</h3>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem;">
                            <div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">Total Spend</div>
                                <div style="font-size: 1.8rem; font-weight: 700;">$${data.totals.spend.toLocaleString()}</div>
                            </div>
                            <div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">Total Sales</div>
                                <div style="font-size: 1.8rem; font-weight: 700;">$${data.totals.sales.toLocaleString()}</div>
                            </div>
                            <div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">ROAS</div>
                                <div style="font-size: 1.8rem; font-weight: 700;">${data.totals.roas.toFixed(2)}x</div>
                            </div>
                        </div>
                    </div>

                    <!-- Issues Breakdown -->
                    <div class="result-card">
                        <h3 style="margin-bottom: 1.5rem;">Where Your Money Is Going</h3>
                        <div class="breakdown-list">
                            ${data.issues.map(issue => `
                                <div class="breakdown-item ${issue.priority}">
                                    <div class="breakdown-header">
                                        <div>
                                            <span style="padding: 0.3rem 0.8rem; background: ${issue.priority === 'high' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(245, 158, 11, 0.1)'}; color: ${issue.priority === 'high' ? '#ef4444' : '#f59e0b'}; border-radius: 100px; font-size: 0.75rem; font-weight: 700; margin-right: 0.8rem;">${issue.priority.toUpperCase()}</span>
                                            <span class="breakdown-title">${issue.title}</span>
                                            <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">${issue.description}</p>
                                            ${issue.count ? `<p style="color: var(--text-muted); font-size: 0.8rem; margin-top: 0.3rem;">${issue.count} items found</p>` : ''}
                                        </div>
                                        <div class="breakdown-amount" style="color: ${issue.type === 'gain' ? '#10b981' : '#ef4444'};">
                                            ${issue.type === 'gain' ? '+' : '-'}$${Math.round(issue.amount).toLocaleString()}/mo
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>

                    <!-- CTA to Full Product -->
                    <div class="upload-cta-card result-card">
                        <div class="upload-cta-content">
                            <div class="upload-icon">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="12" cy="12" r="10"></circle>
                                    <path d="M12 16v-4"></path>
                                    <path d="M12 8h.01"></path>
                                </svg>
                            </div>
                            <div class="upload-cta-text">
                                <h3>Ready to Fix These Issues?</h3>
                                <p>Saddle AdPulse can automatically:</p>
                                <ul class="upload-benefits">
                                    <li>Generate bulk files to add ${data.issues.find(i => i.title.includes('Zero-Conversion'))?.count || 0} negative keywords</li>
                                    <li>Create ${data.issues.find(i => i.title.includes('Harvest'))?.count || 0} exact match harvest campaigns</li>
                                    <li>Optimize bids across all campaigns for target ROAS</li>
                                    <li>Simulate changes before applying</li>
                                    <li>AI analyst to answer performance questions</li>
                                </ul>
                            </div>
                        </div>
                        <button onclick="window.location.href='index.html#pricing'" class="primary-button large" style="width: 100%;">
                            Start Free Trial - Fix These Issues →
                        </button>
                        <p class="upload-disclaimer">14-day free trial • No credit card required • Cancel anytime</p>
                    </div>

                    <!-- Restart -->
                    <div style="text-align: center;">
                        <button onclick="location.reload()" style="background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 0.95rem;">
                            ← Analyze another account
                        </button>
                    </div>
                </div>
            </div>
        </section>
    `;

    // Insert results into page
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = resultsHTML;
    document.querySelector('body').appendChild(tempDiv.firstElementChild);

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function getScoreColor(score) {
    if (score >= 80) return '#10b981';
    if (score >= 60) return '#f59e0b';
    return '#ef4444';
}

document.getElementById('getScoreBtn').addEventListener('click', showQuizResults);

// Initialize on load
initQuiz();
