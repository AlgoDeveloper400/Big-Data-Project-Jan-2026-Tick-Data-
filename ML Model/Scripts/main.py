# main.py
import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
import uvicorn
from typing import Dict

# Import utilities
try:
    from fastapi_utils import load_parameters, get_paths_from_params, validate_parameters, get_trained_symbols, create_response
except ImportError:
    # Create fallback functions if fastapi_utils is not available
    import json
    from pathlib import Path
    from datetime import datetime
    
    def load_parameters(json_path: str) -> Dict:
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def get_paths_from_params(params: Dict) -> Dict[str, Path]:
        paths_section = params.get('paths', {})
        return {
            'artifacts': Path(paths_section.get('artifacts_path', '')),
            'data': Path(paths_section.get('data_path', '')),
            'symbols': Path(paths_section.get('symbols_path', '')),
            'scripts': Path(paths_section.get('scripts_path', ''))
        }
    
    def validate_parameters(params: Dict) -> bool:
        required_sections = ['paths', 'mlflow', 'data', 'model', 'training']
        for section in required_sections:
            if section not in params:
                raise ValueError(f"Missing required section: {section}")
        return True
    
    def get_trained_symbols(artifacts_path: Path) -> list:
        trained_symbols = []
        if artifacts_path.exists():
            for item in artifacts_path.iterdir():
                if item.is_dir():
                    model_path = item / 'best_model.pth'
                    scaler_path = item / 'scaler.pkl'
                    test_data_path = item / 'test_results.json'
                    if model_path.exists() and scaler_path.exists() and test_data_path.exists():
                        trained_symbols.append(item.name)
        return sorted(trained_symbols)
    
    def create_response(success: bool, message: str, data: Dict = None) -> Dict:
        response = {
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        if data:
            response["data"] = data
        return response

# Import training and testing functions
try:
    from train import main_train_val as train_val_main
    TRAIN_VAL_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Train/Val script not available: {e}")
    TRAIN_VAL_AVAILABLE = False

try:
    from test import main_test as test_main
    TEST_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Test script not available: {e}")
    TEST_AVAILABLE = False

# Import live endpoint manager (but don't initialize it yet)
try:
    from live_endpoint import LiveEndpointManager
    LIVE_ENDPOINT_AVAILABLE = True
    live_manager = None  # Initialize as None, will create on demand
except ImportError as e:
    print(f"⚠ Live endpoint not available: {e}")
    LIVE_ENDPOINT_AVAILABLE = False
    live_manager = None

# --- Fixed parameters path ---
PARAMETERS_PATH = r"your\own\path\ML Model\Model Parameters\parameters.json"

# --- FastAPI app ---
app = FastAPI(
    title="Tick Data ML Pipeline",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
    description="Live anomaly detection for tick data during specific time windows"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline_state = {
    "train_val_running": False,
    "train_val_status": "idle",
    "test_running": False,
    "test_status": "idle",
    "live_running": False,
    "live_status": "idle"
}

def ensure_live_manager():
    """Ensure live manager is created when needed"""
    global live_manager
    
    if live_manager is None and LIVE_ENDPOINT_AVAILABLE:
        try:
            live_manager = LiveEndpointManager()
            print("✅ Live endpoint manager initialized (models NOT loaded yet)")
        except Exception as e:
            print(f"❌ Failed to initialize live endpoint manager: {e}")
            return False
    
    return live_manager is not None

# --- Training Functions ---
def run_train_val():
    """Run training and validation"""
    try:
        print("🚀 Starting train_val pipeline...")
        params = load_parameters(PARAMETERS_PATH)
        validate_parameters(params)
        paths = get_paths_from_params(params)
        
        print("📋 Parameters loaded:")
        print(f"  • Epochs: {params['training']['num_epochs']}")
        print(f"  • Batch size: {params['training']['batch_size']}")
        print(f"  • Features: {params['data']['features']}")
        
        result = train_val_main(params, paths)
        return result
        
    except Exception as e:
        print(f"❌ Train/Val error: {e}")
        traceback.print_exc()
        raise

def run_test():
    """Run testing"""
    try:
        print("🚀 Starting test pipeline...")
        params = load_parameters(PARAMETERS_PATH)
        validate_parameters(params)
        paths = get_paths_from_params(params)
        
        trained_symbols = get_trained_symbols(paths['artifacts'])
        if not trained_symbols:
            raise ValueError("No trained models found. Run train_val first.")
        
        print(f"📋 Testing {len(trained_symbols)} trained symbols")
        result = test_main(params, paths)
        return result
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        traceback.print_exc()
        raise

# --- API Endpoints ---
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "Tick Data ML Pipeline",
        "version": "2.0.0",
        "description": "Live anomaly detection during 7:50-8:00 and 13:50-14:00 windows",
        "endpoints": {
            "/train_val": "POST - Start training and validation",
            "/test": "POST - Start testing",
            "/live/start": "POST - Start live endpoint",
            "/live/stop": "POST - Stop live endpoint",
            "/live/health": "GET - Live endpoint health check",
            "/status": "GET - Overall status",
            "/docs": "GET - API documentation"
        },
        "parameters_file": PARAMETERS_PATH
    }

@app.post("/train_val")
def train_val_endpoint(background_tasks: BackgroundTasks):
    """Start training and validation"""
    if not TRAIN_VAL_AVAILABLE:
        return JSONResponse(
            create_response(False, "Training function not available"),
            status_code=500
        )
    
    if pipeline_state["train_val_running"]:
        return JSONResponse(
            create_response(False, "Training is already running"),
            status_code=400
        )
    
    pipeline_state["train_val_running"] = True
    pipeline_state["train_val_status"] = "starting"
    
    def train_val_task():
        try:
            run_train_val()
            pipeline_state["train_val_status"] = "completed"
        except Exception as e:
            pipeline_state["train_val_status"] = f"failed: {str(e)}"
        finally:
            pipeline_state["train_val_running"] = False
    
    background_tasks.add_task(train_val_task)
    
    return create_response(
        True,
        "Training and validation started",
        {
            "status": "started",
            "parameters_file": PARAMETERS_PATH
        }
    )

@app.post("/test")
def test_endpoint(background_tasks: BackgroundTasks):
    """Start testing"""
    if not TEST_AVAILABLE:
        return JSONResponse(
            create_response(False, "Testing function not available"),
            status_code=500
        )
    
    if pipeline_state["test_running"]:
        return JSONResponse(
            create_response(False, "Testing is already running"),
            status_code=400
        )
    
    if pipeline_state["train_val_running"]:
        return JSONResponse(
            create_response(False, "Training is currently running"),
            status_code=400
        )
    
    pipeline_state["test_running"] = True
    pipeline_state["test_status"] = "starting"
    
    def test_task():
        try:
            run_test()
            pipeline_state["test_status"] = "completed"
        except Exception as e:
            pipeline_state["test_status"] = f"failed: {str(e)}"
        finally:
            pipeline_state["test_running"] = False
    
    background_tasks.add_task(test_task)
    
    return create_response(
        True,
        "Testing started",
        {
            "status": "started",
            "parameters_file": PARAMETERS_PATH
        }
    )

@app.post("/live/start")
def live_start_endpoint():
    """Start live endpoint"""
    if not LIVE_ENDPOINT_AVAILABLE:
        return JSONResponse(
            create_response(False, "Live endpoint not available"),
            status_code=500
        )
    
    # Ensure live manager is created
    if not ensure_live_manager():
        return JSONResponse(
            create_response(False, "Failed to initialize live endpoint manager"),
            status_code=500
        )
    
    if pipeline_state["live_running"]:
        return JSONResponse(
            create_response(False, "Live endpoint is already running"),
            status_code=400
        )
    
    try:
        success = live_manager.start()
        
        if not success:
            return JSONResponse(
                create_response(False, "Failed to start live endpoint"),
                status_code=500
            )
        
        pipeline_state["live_running"] = True
        pipeline_state["live_status"] = "running"
        
        return create_response(
            True,
            "Live endpoint started",
            {
                "status": "started",
                "timestamp": datetime.now().isoformat(),
                "message": "Models loaded and monitoring time windows: 7:50-8:00 and 13:50-14:00"
            }
        )
    except Exception as e:
        pipeline_state["live_status"] = f"failed: {str(e)}"
        return JSONResponse(
            create_response(False, f"Failed to start live endpoint: {str(e)}"),
            status_code=500
        )

@app.post("/live/stop")
def live_stop_endpoint():
    """Stop live endpoint"""
    if not LIVE_ENDPOINT_AVAILABLE:
        return JSONResponse(
            create_response(False, "Live endpoint not available"),
            status_code=500
        )
    
    # Ensure live manager is created
    if not ensure_live_manager():
        return JSONResponse(
            create_response(False, "Failed to initialize live endpoint manager"),
            status_code=500
        )
    
    if not pipeline_state["live_running"]:
        return JSONResponse(
            create_response(False, "Live endpoint is not running"),
            status_code=400
        )
    
    try:
        success = live_manager.stop()
        
        if not success:
            return JSONResponse(
                create_response(False, "Failed to stop live endpoint"),
                status_code=500
            )
        
        pipeline_state["live_running"] = False
        pipeline_state["live_status"] = "stopped"
        
        return create_response(
            True,
            "Live endpoint stopped",
            {
                "status": "stopped",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        pipeline_state["live_status"] = f"failed: {str(e)}"
        return JSONResponse(
            create_response(False, f"Failed to stop live endpoint: {str(e)}"),
            status_code=500
        )

@app.get("/live/health")
def live_health_endpoint():
    """Live endpoint health check"""
    if not LIVE_ENDPOINT_AVAILABLE:
        return JSONResponse(
            create_response(False, "Live endpoint not available"),
            status_code=500
        )
    
    # Ensure live manager is created
    if not ensure_live_manager():
        return JSONResponse(
            create_response(False, "Failed to initialize live endpoint manager"),
            status_code=500
        )
    
    try:
        health_data = live_manager.health_check()
        return create_response(
            True,
            "Live endpoint health check",
            health_data
        )
    except Exception as e:
        return JSONResponse(
            create_response(False, f"Health check failed: {str(e)}"),
            status_code=500
        )

@app.get("/status")
def get_status():
    """Get pipeline status"""
    try:
        params = load_parameters(PARAMETERS_PATH)
        paths = get_paths_from_params(params)
        trained_symbols = get_trained_symbols(paths['artifacts'])
        
        live_health = None
        if LIVE_ENDPOINT_AVAILABLE and live_manager:
            live_health = live_manager.health_check()
        
        return create_response(True, "Pipeline status", {
            "train_val": {
                "running": pipeline_state["train_val_running"],
                "status": pipeline_state["train_val_status"]
            },
            "test": {
                "running": pipeline_state["test_running"],
                "status": pipeline_state["test_status"]
            },
            "live": {
                "running": pipeline_state["live_running"],
                "status": pipeline_state["live_status"],
                "health": live_health
            },
            "trained_symbols": trained_symbols,
            "trained_symbols_count": len(trained_symbols),
            "time_windows": ["7:50-8:00", "13:50-14:00"],
            "parameters_file": PARAMETERS_PATH
        })
    except Exception as e:
        return create_response(False, f"Status error: {str(e)}")

# --- Swagger UI ---
@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html():
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Tick Data ML Pipeline",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui.css",
    )
    
    custom_style = """
    <style>
    body { 
        font-family: Arial, sans-serif; 
        background: #f5f5f5; 
        margin: 0;
        padding: 20px;
    }
    .swagger-ui .topbar { 
        display: none; 
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1 {
        color: #333;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 10px;
    }
    .endpoints {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    .endpoint-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
    .method {
        font-weight: bold;
        color: #4CAF50;
    }
    .path {
        font-family: monospace;
        color: #333;
        margin: 5px 0;
    }
    .description {
        color: #666;
        font-size: 14px;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 5px;
    }
    .running { background: #4CAF50; color: white; }
    .stopped { background: #f44336; color: white; }
    .idle { background: #ff9800; color: white; }
    </style>
    
    <div class="container">
        <h1>📊 Tick Data ML Pipeline API</h1>
        <p>Live anomaly detection during specific time windows (7:50-8:00 and 13:50-14:00)</p>
        
        <div class="endpoints">
            <div class="endpoint-card">
                <div class="method">POST</div>
                <div class="path">/train_val</div>
                <div class="description">Start training and validation pipeline</div>
            </div>
            
            <div class="endpoint-card">
                <div class="method">POST</div>
                <div class="path">/test</div>
                <div class="description">Start testing pipeline</div>
            </div>
            
            <div class="endpoint-card">
                <div class="method">POST</div>
                <div class="path">/live/start</div>
                <div class="description">Start live endpoint for anomaly detection</div>
            </div>
            
            <div class="endpoint-card">
                <div class="method">POST</div>
                <div class="path">/live/stop</div>
                <div class="description">Stop live endpoint</div>
            </div>
            
            <div class="endpoint-card">
                <div class="method">GET</div>
                <div class="path">/live/health</div>
                <div class="description">Check live endpoint health status</div>
            </div>
            
            <div class="endpoint-card">
                <div class="method">GET</div>
                <div class="path">/status</div>
                <div class="description">Get overall pipeline status</div>
            </div>
        </div>
    </div>
    """
    
    return HTMLResponse(html.body.decode("utf-8") + custom_style)

# --- Run ---
if __name__ == "__main__":
    print("=" * 80)
    print("TICK DATA ML PIPELINE API")
    print("=" * 80)
    print(f"Parameters file: {PARAMETERS_PATH}")
    print("Time windows: 7:50-8:00 and 13:50-14:00")
    print("=" * 80)
    print("Endpoints:")
    print("  POST /train_val    - Start training")
    print("  POST /test         - Start testing")
    print("  POST /live/start   - Start live endpoint")
    print("  POST /live/stop    - Stop live endpoint")
    print("  GET  /live/health  - Live health check")
    print("  GET  /status       - Check overall status")
    print("  GET  /docs         - API documentation")
    print("=" * 80)
    print("API: http://127.0.0.1:9935")
    print("Docs: http://127.0.0.1:9935/docs")
    print("=" * 80)
    
    if not os.path.exists(PARAMETERS_PATH):
        print(f"❌ ERROR: Parameters file not found!")
        print(f"Expected at: {PARAMETERS_PATH}")
        exit(1)
    
    try:
        params = load_parameters(PARAMETERS_PATH)
        validate_parameters(params)
        paths = get_paths_from_params(params)
        
        print("✅ Configuration loaded successfully")
        print(f"📁 Artifacts path: {paths['artifacts']}")
        print(f"📁 Data path: {paths['data']}")
        print(f"📁 Symbols path: {paths['symbols']}")
        
        trained_symbols = get_trained_symbols(paths['artifacts'])
        print(f"📊 Trained symbols available: {len(trained_symbols)} symbols")
        
        if LIVE_ENDPOINT_AVAILABLE:
            print(f"📡 Live endpoint: Ready (models will load when started)")
        else:
            print(f"⚠ Live endpoint: Not available")
        
        print("=" * 80)
        print("ℹ️  Note: Live endpoint models will ONLY load when /live/start is called")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        exit(1)
    
    uvicorn.run("main:app", host="127.0.0.1", port=9935, reload=False)