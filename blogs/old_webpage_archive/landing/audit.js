// ========================================
// SADDLE ADPULSE - IMPACT READINESS ASSESSMENT
// Measures impact clarity, not optimization quality
// ========================================

// Quiz questions - Impact Readiness Assessment
const questions = [
    {
        id: 'spend',
        question: "Roughly how much do you spend on Amazon ads per month?",
        options: [
            { value: 'under-2k', label: 'Less than $2k', baseSpend: 1500 },
            { value: '2k-10k', label: '$2k–10k', baseSpend: 6000 },
            { value: '10k-50k', label: '$10k–50k', baseSpend: 30000 },
            { value: '50k+', label: '$50k+', baseSpend: 75000 }
        ]
    },
    {
        id: 'frequency',
        question: "How often do you make changes to your Amazon ads?",
        options: [
            { value: 'daily', label: 'Almost daily', points: 20 },
            { value: 'few-times-week', label: 'A few times a week', points: 16 },
            { value: 'once-week', label: 'About once a week', points: 12 },
            { value: 'less-often', label: 'Less often', points: 8 }
        ]
    },
    {
        id: 'attribution',
        question: "When results change, which of these feels closest to how you usually explain it?",
        options: [
            { value: 'our-changes', label: '"Our changes worked — the metrics improved"', points: 10 },
            { value: 'market-moved', label: '"The market moved (seasonality, demand, competition)"', points: 6 },
            { value: 'mix-unsure', label: '"It\'s usually a mix, but I can\'t tell how much is which"', points: 14 },
            { value: 'not-sure', label: '"Honestly, I\'m not sure"', points: 4 }
        ]
    },
    {
        id: 'confidence',
        question: 'If someone asked, "Did your recent ad changes actually add value?", how confidently could you answer?',
        options: [
            { value: 'confident', label: 'I could answer that confidently', points: 30 },
            { value: 'rough', label: "I'd give a rough explanation", points: 18 },
            { value: 'struggle', label: "I'd struggle to answer clearly", points: 8 },
            { value: 'no-idea', label: "I wouldn't know how to answer", points: 4 }
        ]
    },
    {
        id: 'communication',
        question: "How do you usually talk about ad performance today?",
        options: [
            { value: 'metrics', label: 'Mostly through metrics (ROAS, CPC, ACOS, etc.)', points: 12 },
            { value: 'trends', label: 'By explaining trends over time', points: 14 },
            { value: 'context', label: 'By giving context without clear numbers', points: 10 },
            { value: 'struggle', label: 'I struggle to explain it clearly', points: 6 }
        ]
    }
];

let quizAnswers = {};
let selectedFile = null;
let quizInitialized = false;

// Initialize quiz
function initQuiz() {
    const container = document.getElementById('questionsContainer');
    container.innerHTML = ''; // Clear existing content

    questions.forEach((q, idx) => {
        const questionDiv = document.createElement('div');
        questionDiv.className = 'question-card';
        questionDiv.innerHTML = `
            <div class="question-header">
                <div class="question-number">${idx + 1}</div>
                <div class="question-content">
                    <h3 class="question-text">${q.question}</h3>
                    <div class="question-checkmark" id="check-${q.id}">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                    </div>
                </div>
            </div>
            <div class="options-modern" id="options-${q.id}">
                ${q.options.map(opt => `
                    <button
                        type="button"
                        class="option-button"
                        data-question="${q.id}"
                        data-value="${opt.value}"
                        onclick='selectAnswer("${q.id}", ${JSON.stringify(opt).replace(/'/g, "&apos;")})'
                    >
                        <span class="option-text">${opt.label}</span>
                        <span class="option-check">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="9 12 11 14 15 10"></polyline>
                            </svg>
                        </span>
                    </button>
                `).join('')}
            </div>
        `;
        container.appendChild(questionDiv);
    });
}

function selectAnswer(questionId, option) {
    // Store answer
    quizAnswers[questionId] = option;

    // Update UI - option buttons
    const options = document.querySelectorAll(`[data-question="${questionId}"]`);
    options.forEach(opt => {
        if (opt.dataset.value === option.value) {
            opt.classList.add('selected');
        } else {
            opt.classList.remove('selected');
        }
    });

    // Show checkmark on question card
    const checkmark = document.getElementById(`check-${questionId}`);
    if (checkmark) {
        checkmark.classList.add('visible');
    }

    // Update progress
    const answeredCount = Object.keys(quizAnswers).length;
    const progress = (answeredCount / questions.length) * 100;

    // Update progress bar
    const progressBar = document.getElementById('progressFill');
    if (progressBar) {
        progressBar.style.width = progress + '%';
    }

    // Update progress counter
    const progressCounter = document.getElementById('progressPercent');
    if (progressCounter) {
        progressCounter.textContent = answeredCount;
    }

    // Enable button when all answered
    const submitBtn = document.getElementById('getScoreBtn');
    if (submitBtn) {
        submitBtn.disabled = answeredCount < questions.length;
        if (answeredCount === questions.length) {
            submitBtn.classList.add('ready');
        } else {
            submitBtn.classList.remove('ready');
        }
    }
}

function calculateImpactScore() {
    // A) Decision Activity (Q2) — max 20 pts
    const activityPoints = quizAnswers.frequency?.points || 8;

    // B) Explanation Method (Q3) — max 30 pts
    const explanationPoints = quizAnswers.attribution?.points || 4;

    // C) Confidence Under Scrutiny (Q4) — max 30 pts
    const confidencePoints = quizAnswers.confidence?.points || 4;

    // D) Communication Style (Q5) — max 20 pts
    const communicationPoints = quizAnswers.communication?.points || 6;

    // Calculate total score
    let totalScore = activityPoints + explanationPoints + confidencePoints + communicationPoints;

    // Clamp score between 20 and 90
    totalScore = Math.max(20, Math.min(90, totalScore));

    // Get base spend from Q1
    const baseSpend = quizAnswers.spend?.baseSpend || 6000;

    // Calculate Clarity Factor
    const clarityFactor = (100 - totalScore) / 100;

    // Calculate Unexplained Value Range
    const lowValue = Math.round(baseSpend * clarityFactor * 0.15);
    const highValue = Math.round(baseSpend * clarityFactor * 0.30);

    // Determine limiting factors based on answers
    const limitingFactors = [];

    // Check communication style
    if (quizAnswers.communication?.value === 'metrics' ||
        quizAnswers.communication?.value === 'context' ||
        quizAnswers.communication?.value === 'struggle') {
        limitingFactors.push('Results are evaluated using metrics and trends');
    }

    // Check attribution understanding
    if (quizAnswers.attribution?.value === 'our-changes' ||
        quizAnswers.attribution?.value === 'mix-unsure') {
        limitingFactors.push("It's hard to separate your actions from market changes");
    }

    // Check confidence level
    if (quizAnswers.confidence?.value === 'struggle' ||
        quizAnswers.confidence?.value === 'no-idea' ||
        quizAnswers.confidence?.value === 'rough') {
        limitingFactors.push('Performance explanations rely on intuition rather than evidence');
    }

    // Ensure we have exactly 3 factors, add fallback if needed
    if (limitingFactors.length < 3) {
        const fallbackFactors = [
            'Results are evaluated using metrics and trends',
            "It's hard to separate your actions from market changes",
            'Performance explanations rely on intuition rather than evidence'
        ];
        fallbackFactors.forEach(factor => {
            if (limitingFactors.length < 3 && !limitingFactors.includes(factor)) {
                limitingFactors.push(factor);
            }
        });
    }

    return {
        score: totalScore,
        lowValue: lowValue,
        highValue: highValue,
        limitingFactors: limitingFactors.slice(0, 3)
    };
}

function showQuizResults() {
    const results = calculateImpactScore();

    // Hide quiz, show results
    document.getElementById('quizSection').classList.add('hidden');
    document.getElementById('resultsSection').classList.remove('hidden');

    // Update score
    const healthScoreEl = document.getElementById('healthScore');
    healthScoreEl.textContent = results.score;

    // Add color class based on score
    if (results.score >= 70) {
        healthScoreEl.style.color = '#0891B2'; // Brand teal - good impact clarity
    } else if (results.score >= 50) {
        healthScoreEl.style.color = '#D4A574'; // Muted amber - moderate clarity
    } else {
        healthScoreEl.style.color = '#C27563'; // Muted terracotta - low clarity
    }

    // Update opportunity (now "unexplained value")
    document.getElementById('opportunityAmount').textContent =
        `$${results.lowValue.toLocaleString()} – $${results.highValue.toLocaleString()}`;

    // Update breakdown with limiting factors
    const breakdownHTML = results.limitingFactors.map(factor => `
        <div class="breakdown-item">
            <div class="breakdown-header">
                <span class="breakdown-title">${factor}</span>
            </div>
        </div>
    `).join('');
    document.getElementById('breakdownList').innerHTML = breakdownHTML;

    // Scroll modal to top
    const modalContainer = document.querySelector('.modal-container');
    if (modalContainer) {
        modalContainer.scrollTop = 0;
    }
}

// Initialize audit quiz when called
function initializeAuditQuiz() {
    if (!quizInitialized) {
        initQuiz();
        quizInitialized = true;
    }
}

// Get Score button handler
const getScoreBtn = document.getElementById('getScoreBtn');
if (getScoreBtn) {
    getScoreBtn.addEventListener('click', showQuizResults);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeAuditQuiz);
} else {
    initializeAuditQuiz();
}

// ========================================
// FILE UPLOAD FUNCTIONALITY (PRESERVED)
// Note: This section remains for future CSV upload feature
// Currently not used in Impact Readiness Assessment
// ========================================

// File upload handlers
const fileInput = document.getElementById('csvUpload');
const fileLabel = document.getElementById('fileLabel');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const analyzeBtn = document.getElementById('analyzeBtn');

if (fileInput) {
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            selectedFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
            fileLabel.classList.add('file-selected');
            analyzeBtn.disabled = false;
        }
    });
}

// Note: CSV analysis functionality preserved but not currently used
// Can be reactivated when file upload becomes part of the flow
