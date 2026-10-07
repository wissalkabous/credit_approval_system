"""
Application FastAPI - API REST + Interface Web
"""
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pathlib import Path
# Import chatbot logic built earlier (chatbot/chatbot.py)
from chatbot.chatbot import ask_chatbot
from api.schemas import PredictionInput, PredictionResponse, HealthResponse, ChatRequest, ChatResponse




# Force UTF-8 on Windows so emoji in logs don't crash the server
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))


MODEL_PATH = "models/credit_model.pkl"       # full path for existence check
MODEL_FILE = "credit_model.pkl"              # filename only — repository prepends "models/"
MODEL_INFO_PATH = "models/credit_model_info.json"
DATA_PATH = "data/raw/credit_data.csv"

# Maps snake_case API fields -> PascalCase CSV columns
FEATURE_MAP = {
    'age': 'Age',
    'monthly_income': 'MonthlyIncome',
    'employment_status': 'EmploymentStatus',
    'employment_duration': 'EmploymentDuration',
    'total_debt': 'TotalDebt',
    'loan_amount': 'LoanAmount',
    'loan_duration': 'LoanDuration',
    'credit_score': 'CreditScore',
    'repayment_history': 'RepaymentHistory',
    'num_late_payments': 'NumLatePayments',
    'num_open_accounts': 'NumOpenAccounts',
    'account_age_years': 'AccountAgeYears',
}

predictor = None


# ── Async Programming: background task (ne bloque pas la réponse HTTP) ──
async def log_prediction_async(statut: str, confiance: str, risque: str):
    """
    Asynchronous Programming: log la prédiction en tâche de fond.
    La réponse est renvoyée au client AVANT que ce log soit écrit.
    """
    await asyncio.sleep(0)  # yield au event loop — libère le thread
    print(f"[ASYNC LOG] Decision: {statut} | Confidence: {confiance} | Risk: {risque}")


def init_predictor():
    global predictor

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}. Run: python train.py")
        return False

    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        return False

    try:
        container = initialize_container()
        data_processor = container.create_data_processor(DATA_PATH)
        data_processor.load()
        data_processor.clean()
        data_processor.encode()
        data_processor.prepare()

        model = container.create_model(model_type="random_forest")
        model.load(MODEL_FILE)

        predictor = container.create_predictor(model, data_processor)
        print("Predictor initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("Credit Approval System v2.0 - Starting")
    print("=" * 60)
    init_predictor()
    yield


app = FastAPI(
    title="Credit Approval System",
    description="Bank credit decision support powered by Machine Learning",
    version="2.0.0",
    lifespan=lifespan,
)


# ============= ROUTES =============

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    return get_html_interface()


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    status = "ok" if predictor else "model_not_loaded"
    return {
        "status": status,
        "message": "API is running" if predictor else "Model not loaded. Run: python train.py"
    }


@app.get("/api/model-info")
async def get_model_info():
    if not os.path.exists(MODEL_INFO_PATH):
        raise HTTPException(status_code=404, detail="Model info not found")
    return load_json(MODEL_INFO_PATH)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(data: PredictionInput, background_tasks: BackgroundTasks):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python train.py")

    try:
        client_data = {FEATURE_MAP[k]: v for k, v in data.model_dump().items()}
        prediction, probability = predictor.predict_single(client_data)
        result = predictor.format_result(prediction, probability, client_data)

        # Async Programming: log en arrière-plan APRÈS avoir renvoyé la réponse
        background_tasks.add_task(
            log_prediction_async,
            result["statut"],
            result["confiance_pct"],
            result["risque"]
        )

        return {
            "statut": result["statut"],
            "prediction": result["prediction"],
            "confiance_pct": result["confiance_pct"],
            "risque": result["risque"],
            "data_client": result["data_client"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(data: ChatRequest):
    """
    Chat endpoint: receives a user question, retrieves relevant
    context from the bank FAQ knowledge base, and returns an
    AI-generated answer.
    """
    try:
        answer = ask_chatbot(data.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= HTML INTERFACE =============

def get_html_interface() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Approval System</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{
            font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;
            background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
            min-height:100vh;
            padding:30px 20px;
        }
        .container{
            background:#fff;
            border-radius:20px;
            box-shadow:0 25px 80px rgba(0,0,0,.35);
            max-width:920px;
            margin:0 auto;
            overflow:hidden;
        }
        .header{
            background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
            color:#fff;
            padding:38px 40px 30px;
            position:relative;
            overflow:hidden;
        }
        .header::before{
            content:'';position:absolute;top:-60px;right:-40px;
            width:280px;height:280px;
            background:rgba(255,255,255,.06);border-radius:50%;
        }
        .header::after{
            content:'';position:absolute;bottom:-80px;right:80px;
            width:180px;height:180px;
            background:rgba(255,255,255,.04);border-radius:50%;
        }
        .header-title{font-size:28px;font-weight:700;margin-bottom:6px;position:relative;text-align:center}
        .header-sub{font-size:13px;opacity:.85;position:relative;text-align:center}
        .header-badges{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap;position:relative;justify-content:center}
        .badge{
            background:rgba(255,255,255,.18);
            padding:4px 13px;border-radius:20px;
            font-size:12px;font-weight:500;
            backdrop-filter:blur(4px);
        }
        .form-content{padding:32px 40px}
        .section{margin-bottom:26px}
        .section-title{
            font-size:11px;font-weight:700;text-transform:uppercase;
            letter-spacing:1.2px;color:#764ba2;
            margin-bottom:14px;padding-bottom:8px;
            border-bottom:2px solid #f0ebfa;
            display:flex;align-items:center;gap:7px;
        }
        .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
        .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
        .grid-4{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px}
        .form-group{display:flex;flex-direction:column}
        .form-group label{
            font-size:11px;font-weight:700;color:#555;
            margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px;
        }
        .form-group input,.form-group select{
            padding:10px 12px;
            border:2px solid #e8e0f5;border-radius:10px;
            font-size:14px;color:#333;
            transition:all .25s;background:#fafafa;
        }
        .form-group input:focus,.form-group select:focus{
            outline:none;border-color:#764ba2;
            background:#fff;box-shadow:0 0 0 3px rgba(118,75,162,.12);
        }
        .input-hint{font-size:10px;color:#aaa;margin-top:4px}
        .dti-box{
            background:#f8f4ff;border:1px solid #e0d0f5;
            border-radius:10px;padding:12px 16px;
            margin-top:12px;display:flex;align-items:center;justify-content:space-between;gap:16px;
        }
        .dti-left{flex:1}
        .dti-label{font-size:12px;font-weight:600;color:#555;margin-bottom:6px}
        .dti-bar-wrap{height:6px;background:#e0e0e0;border-radius:3px;overflow:hidden}
        .dti-bar{height:100%;border-radius:3px;transition:width .4s,background .4s;width:0%}
        .dti-val{font-size:20px;font-weight:800;min-width:52px;text-align:right}
        .submit-btn{
            width:100%;padding:15px;
            background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
            color:#fff;border:none;border-radius:12px;
            font-size:16px;font-weight:700;cursor:pointer;
            transition:all .3s;margin-top:8px;letter-spacing:.4px;
        }
        .submit-btn:hover{transform:translateY(-2px);box-shadow:0 12px 30px rgba(102,126,234,.4)}
        .submit-btn:active{transform:translateY(0)}
        .submit-btn:disabled{opacity:.65;cursor:not-allowed;transform:none}
        .loading{display:none;text-align:center;padding:22px}
        .loading.show{display:block}
        .spinner{
            border:4px solid #f0ebfa;border-top:4px solid #764ba2;
            border-radius:50%;width:44px;height:44px;
            animation:spin .9s linear infinite;margin:0 auto 12px;
        }
        @keyframes spin{to{transform:rotate(360deg)}}
        .error{
            display:none;background:#fff0f0;color:#c0392b;
            padding:14px 18px;border-radius:10px;
            border-left:4px solid #e74c3c;
            margin-top:18px;font-size:14px;
        }
        .error.show{display:block}
        .result{
            display:none;margin-top:24px;
            border-radius:14px;overflow:hidden;
            animation:slideIn .35s ease-out;
        }
        .result.show{display:block}
        @keyframes slideIn{
            from{opacity:0;transform:translateY(14px)}
            to{opacity:1;transform:translateY(0)}
        }
        .result-header{
            padding:22px 26px;
            display:flex;align-items:center;justify-content:space-between;
        }
        .result.approved .result-header{
            background:linear-gradient(135deg,#1de9b6 0%,#1565c0 100%);color:#fff;
        }
        .result.rejected .result-header{
            background:linear-gradient(135deg,#e53935 0%,#b71c1c 100%);color:#fff;
        }
        .result-status{font-size:20px;font-weight:800}
        .result-conf{font-size:34px;font-weight:900}
        .result-body{
            background:#fff;padding:20px 26px;
            border:1px solid #eee;border-top:none;
            border-radius:0 0 14px 14px;
            display:grid;grid-template-columns:210px 1fr;gap:20px;align-items:center;
        }
        .gauge-wrap{display:flex;flex-direction:column;align-items:center}
        .gauge-svg{width:100%;max-width:210px}
        #gaugeTrack{stroke:#e8e0f5}
        #gaugeArc{
            stroke-dasharray:251.33;stroke-dashoffset:251.33;
            transition:stroke-dashoffset .9s ease,stroke .3s;
        }
        .metrics-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
        .metric{
            text-align:center;padding:13px 8px;
            background:#f9f9f9;border-radius:10px;
        }
        .metric-label{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:#999;margin-bottom:5px}
        .metric-value{font-size:15px;font-weight:700;color:#333}
        @media(max-width:700px){
            .grid-3,.grid-4{grid-template-columns:1fr 1fr}
            .result-body{grid-template-columns:1fr}
            .form-content{padding:22px 18px}
            .header{padding:28px 20px}
        }
        @media(max-width:480px){
            .grid-3,.grid-4,.grid-2{grid-template-columns:1fr}
        }


        


        /* ============= CHAT WIDGET ============= */







/* Floating button, fixed to bottom-right corner of the screen */
.chat-toggle{
    position:fixed;bottom:24px;right:24px;
    width:58px;height:58px;border-radius:50%;
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    color:#fff;border:none;cursor:pointer;
    font-size:26px;display:flex;align-items:center;justify-content:center;
    box-shadow:0 8px 24px rgba(118,75,162,.45);
    transition:transform .25s;
    z-index:1000;
}
.chat-toggle:hover{transform:scale(1.08)}

/* The chat panel itself — hidden by default */
.chat-panel{
    position:fixed;bottom:94px;right:24px;
    width:340px;max-width:90vw;height:440px;
    background:#fff;border-radius:16px;
    box-shadow:0 20px 60px rgba(0,0,0,.3);
    display:none;flex-direction:column;overflow:hidden;
    z-index:1000;
}
.chat-panel.show{display:flex}

.chat-header{
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    color:#fff;padding:14px 18px;font-weight:700;font-size:14px;
    display:flex;align-items:center;justify-content:space-between;
}
.chat-close{background:none;border:none;color:#fff;font-size:18px;cursor:pointer;opacity:.85}
.chat-close:hover{opacity:1}

.chat-messages{
    flex:1;overflow-y:auto;padding:14px;
    display:flex;flex-direction:column;gap:10px;
    background:#faf9fc;
}
.chat-msg{
    max-width:82%;padding:9px 13px;border-radius:12px;
    font-size:13px;line-height:1.4;
}
.chat-msg.user{
    align-self:flex-end;background:#764ba2;color:#fff;
    border-bottom-right-radius:3px;
}
.chat-msg.bot{
    align-self:flex-start;background:#f0ebfa;color:#333;
    border-bottom-left-radius:3px;white-space:pre-wrap;
}
.chat-msg.loading{align-self:flex-start;color:#999;font-style:italic}

.chat-input-row{
    display:flex;padding:10px;gap:8px;
    border-top:1px solid #eee;background:#fff;
}
.chat-input-row input{
    flex:1;padding:9px 12px;border:2px solid #e8e0f5;
    border-radius:20px;font-size:13px;outline:none;
}
.chat-input-row input:focus{border-color:#764ba2}
.chat-send{
    width:36px;height:36px;border-radius:50%;border:none;
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    color:#fff;cursor:pointer;font-size:15px;
}


        



    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="header-title">Credit Approval System</div>
        <div class="header-sub">Bank Credit Decision Support &mdash; Machine Learning Powered</div>
    </div>

    <div class="form-content">
        <form id="form">

            <!-- Section 1: Personal -->
            <div class="section">
                <div class="section-title">&#x1F464; Personal Information</div>
                <div class="grid-3">
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" id="age" min="18" max="75" value="35" required>
                        <span class="input-hint">18 &ndash; 75 years</span>
                    </div>
                    <div class="form-group">
                        <label>Employment Status</label>
                        <select id="employment_status" required>
                            <option value="0">&#x1F4BC; Employed</option>
                            <option value="1">&#x1F3E2; Self-Employed</option>
                            <option value="2">&#x1F3C5; Retired</option>
                            <option value="3">&#x26A0;&#xFE0F; Unemployed</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Employment Duration</label>
                        <input type="number" id="employment_duration" min="0" max="40" value="5" required>
                        <span class="input-hint">Years at current employer</span>
                    </div>
                </div>
            </div>

            <!-- Section 2: Financial -->
            <div class="section">
                <div class="section-title">&#x1F4B0; Financial Situation</div>
                <div class="grid-3">
                    <div class="form-group">
                        <label>Monthly Income (MAD)</label>
                        <input type="number" id="monthly_income" min="3000" step="500" value="5000" required oninput="updateDTI()">
                        <span class="input-hint">Net monthly salary in Dirhams</span>
                    </div>
                    <div class="form-group">
                        <label>Total Existing Debt (MAD)</label>
                        <input type="number" id="total_debt" min="0" step="1000" value="20000" required oninput="updateDTI()">
                        <span class="input-hint">All current obligations</span>
                    </div>
                    <div class="form-group">
                        <label>Credit Score</label>
                        <input type="number" id="credit_score" min="300" max="850" value="700" required>
                        <span class="input-hint">300 (Poor) &ndash; 850 (Excellent)</span>
                    </div>
                </div>
                <div class="dti-box">
                    <div class="dti-left">
                        <div class="dti-label">Debt-to-Income Ratio (DTI)</div>
                        <div class="dti-bar-wrap"><div class="dti-bar" id="dtiBar"></div></div>
                    </div>
                    <div class="dti-val" id="dtiVal" style="color:#27ae60">19%</div>
                </div>
            </div>

            <!-- Section 3: Loan -->
            <div class="section">
                <div class="section-title">&#x1F3E6; Loan Request</div>
                <div class="grid-2">
                    <div class="form-group">
                        <label>Loan Amount (MAD)</label>
                        <input type="number" id="loan_amount" min="5000" step="5000" value="80000" required>
                        <span class="input-hint">Requested amount in Dirhams</span>
                    </div>
                    <div class="form-group">
                        <label>Loan Duration</label>
                        <select id="loan_duration" required>
                            <option value="12">12 months &mdash; 1 year</option>
                            <option value="24">24 months &mdash; 2 years</option>
                            <option value="36" selected>36 months &mdash; 3 years</option>
                            <option value="48">48 months &mdash; 4 years</option>
                            <option value="60">60 months &mdash; 5 years</option>
                            <option value="84">84 months &mdash; 7 years</option>
                            <option value="120">120 months &mdash; 10 years</option>
                            <option value="180">180 months &mdash; 15 years</option>
                            <option value="240">240 months &mdash; 20 years</option>
                            <option value="360">360 months &mdash; 30 years</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Section 4: History -->
            <div class="section">
                <div class="section-title">&#x1F4CA; Account &amp; Credit History</div>
                <div class="grid-4">
                    <div class="form-group">
                        <label>Repayment History</label>
                        <select id="repayment_history" required>
                            <option value="1">Poor</option>
                            <option value="4">Average</option>
                            <option value="7" selected>Good</option>
                            <option value="10">Very Good</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Late Payments</label>
                        <input type="number" id="num_late_payments" min="0" max="20" value="1" required>
                        <span class="input-hint">Last 2 years</span>
                    </div>
                    <div class="form-group">
                        <label>Open Accounts</label>
                        <input type="number" id="num_open_accounts" min="0" max="15" value="3" required>
                        <span class="input-hint">Active credit lines</span>
                    </div>
                    <div class="form-group">
                        <label>Account Age</label>
                        <input type="number" id="account_age_years" min="0" max="30" value="5" required>
                        <span class="input-hint">Years with this bank</span>
                    </div>
                </div>
            </div>

            <button type="submit" class="submit-btn" id="submitBtn">
                &#x1F50D; Analyze Credit Application
            </button>
        </form>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="color:#764ba2;font-weight:600;font-size:14px">Analyzing application&hellip;</p>
        </div>

        <div class="error" id="errorBox"></div>

        <div class="result" id="result">
            <div class="result-header">
                <div class="result-status" id="rStatus"></div>
                <div class="result-conf" id="rConf"></div>
            </div>
            <div class="result-body">
                <div class="gauge-wrap">
                    <svg viewBox="0 0 200 115" class="gauge-svg">
                        <path id="gaugeTrack" d="M 20 100 A 80 80 0 0 0 180 100" fill="none" stroke-width="14" stroke-linecap="round"/>
                        <path id="gaugeArc"   d="M 20 100 A 80 80 0 0 0 180 100" fill="none" stroke="#27ae60" stroke-width="14" stroke-linecap="round"/>
                        <text id="gaugeText" x="100" y="83" text-anchor="middle" font-size="26" font-weight="800" fill="#333">--</text>
                        <text x="100" y="101" text-anchor="middle" font-size="9" fill="#bbb" letter-spacing="2">CONFIDENCE</text>
                    </svg>
                </div>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-label">Decision</div>
                        <div class="metric-value" id="rDecision"></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Risk Level</div>
                        <div class="metric-value" id="rRisk"></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">DTI Ratio</div>
                        <div class="metric-value" id="rDTI"></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Loan / Income</div>
                        <div class="metric-value" id="rLTI"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Floating chat button -->
<button class="chat-toggle" id="chatToggle">&#x1F916;</button>

<!-- Chat panel -->
<div class="chat-panel" id="chatPanel">
    <div class="chat-header">
        <span>&#x1F916; Credit Assistant</span>
        <button class="chat-close" id="chatClose">&times;</button>
    </div>
    <div class="chat-messages" id="chatMessages">
        <div class="chat-msg bot">Hi! Ask me anything about loans, eligibility, or the approval process.</div>
    </div>
    <div class="chat-input-row">
        <input type="text" id="chatInput" placeholder="Ask a question...">
        <button class="chat-send" id="chatSend">&#x27A4;</button>
    </div>
</div>


<script>
const GAUGE_LEN = Math.PI * 80; // 251.33 — full semicircle arc length

function animateGauge(probability, isApproved) {
    const color = isApproved ? '#00c896' : '#e53935';
    const arc = document.getElementById('gaugeArc');
    const txt = document.getElementById('gaugeText');
    const p = Math.min(0.998, Math.max(0.002, probability));
    // Reset without transition, then re-enable it to trigger animation
    arc.style.transition = 'none';
    arc.style.strokeDashoffset = GAUGE_LEN;
    arc.getBoundingClientRect(); // force reflow
    arc.style.transition = 'stroke-dashoffset .9s ease, stroke .3s';
    arc.style.stroke = color;
    arc.style.strokeDashoffset = GAUGE_LEN * (1 - p);
    txt.textContent = Math.round(p * 100) + '%';
    txt.style.fill = color;
}

function updateDTI(){
    const inc = parseFloat(document.getElementById('monthly_income').value)||0;
    const debt= parseFloat(document.getElementById('total_debt').value)||0;
    if(inc<=0)return;
    const dti=(debt/(inc*12))*100;
    const v=Math.min(dti,100).toFixed(1);
    const color=dti>50?'#e74c3c':dti>30?'#f39c12':'#27ae60';
    document.getElementById('dtiVal').textContent=v+'%';
    document.getElementById('dtiVal').style.color=color;
    document.getElementById('dtiBar').style.width=Math.min(dti,100)+'%';
    document.getElementById('dtiBar').style.background=color;
}

const form=document.getElementById('form');
const loading=document.getElementById('loading');
const errorBox=document.getElementById('errorBox');
const result=document.getElementById('result');
const submitBtn=document.getElementById('submitBtn');

form.addEventListener('submit',async(e)=>{
    e.preventDefault();
    const g=id=>document.getElementById(id).value;
    const data={
        age:parseInt(g('age')),
        monthly_income:parseFloat(g('monthly_income')),
        employment_status:parseInt(g('employment_status')),
        employment_duration:parseInt(g('employment_duration')),
        total_debt:parseFloat(g('total_debt')),
        loan_amount:parseFloat(g('loan_amount')),
        loan_duration:parseInt(g('loan_duration')),
        credit_score:parseInt(g('credit_score')),
        repayment_history:parseInt(g('repayment_history')),
        num_late_payments:parseInt(g('num_late_payments')),
        num_open_accounts:parseInt(g('num_open_accounts')),
        account_age_years:parseInt(g('account_age_years'))
    };

    loading.classList.add('show');
    errorBox.classList.remove('show');
    result.classList.remove('show','approved','rejected');
    submitBtn.disabled=true;
    // Reset gauge to idle state
    const gaugeArc=document.getElementById('gaugeArc');
    const gaugeText=document.getElementById('gaugeText');
    gaugeArc.style.transition='none';
    gaugeArc.style.strokeDashoffset=GAUGE_LEN;
    gaugeArc.style.stroke='#e8e0f5';
    gaugeText.textContent='--';
    gaugeText.style.fill='#333';

    try{
        const resp=await fetch('/api/predict',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify(data)
        });
        if(!resp.ok){const e=await resp.json();throw new Error(e.detail||'Prediction failed');}
        const p=await resp.json();
        const ok=p.prediction===1;
        const dti=((data.total_debt/(data.monthly_income*12))*100).toFixed(1);
        const lti=((data.loan_amount/(data.monthly_income*12))*100).toFixed(1);
        const probVal=parseFloat(p.confiance_pct)/100;
        animateGauge(ok?probVal:(1-probVal),ok);

        document.getElementById('rStatus').textContent=ok?'APPROVED':'REJECTED';
        document.getElementById('rConf').textContent=p.confiance_pct;
        document.getElementById('rDecision').textContent=ok?'Credit Approved':'Credit Refused';
        document.getElementById('rRisk').textContent=p.risque;
        document.getElementById('rDTI').textContent=dti+'%';
        document.getElementById('rLTI').textContent=lti+'%';

        result.classList.add('show',ok?'approved':'rejected');
        result.scrollIntoView({behavior:'smooth',block:'nearest'});
    }catch(err){
        errorBox.textContent='Error: '+err.message;
        errorBox.classList.add('show');
    }finally{
        loading.classList.remove('show');
        submitBtn.disabled=false;
    }
});

updateDTI();



// ============= CHAT WIDGET LOGIC =============



const chatToggle = document.getElementById('chatToggle');
const chatPanel = document.getElementById('chatPanel');
const chatClose = document.getElementById('chatClose');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const chatSend = document.getElementById('chatSend');

// Show/hide the panel
chatToggle.addEventListener('click', () => chatPanel.classList.toggle('show'));
chatClose.addEventListener('click', () => chatPanel.classList.remove('show'));

// Adds a message bubble to the chat window
function addMessage(text, sender) {
    const msg = document.createElement('div');
    msg.className = 'chat-msg ' + sender;
    msg.textContent = text;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight; // auto-scroll to latest
    return msg;
}

// Sends the question to our /api/chat endpoint and displays the answer
async function sendChatMessage() {
    const question = chatInput.value.trim();
    if (!question) return;

    addMessage(question, 'user');
    chatInput.value = '';

    const loadingMsg = addMessage('Thinking...', 'loading');

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        if (!resp.ok) {
            const e = await resp.json();
            throw new Error(e.detail || 'Chat request failed');
        }

        const data = await resp.json();
        loadingMsg.remove();
        addMessage(data.answer, 'bot');

    } catch (err) {
        loadingMsg.remove();
        addMessage('Sorry, something went wrong: ' + err.message, 'bot');
    }
}

chatSend.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChatMessage();
});


</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
