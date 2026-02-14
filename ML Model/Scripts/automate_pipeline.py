"""
Script to automatically run the entire ML pipeline
1. Checks if FastAPI is running
2. Runs training and validation
3. Runs testing
4. Automatically starts the live endpoint after successful testing
5. Monitors progress
"""
import time
import requests
import sys
import os
from pathlib import Path
import json

class PipelineAutomator:
    def __init__(self):
        self.fastapi_url = "http://127.0.0.1:9935"
        self.params_path = r"your\own\path\ML Model\Model Parameters\parameters.json"
        self.auto_start_live = True  # Set to False if you want manual control
        
    def check_fastapi_running(self):
        """Check if FastAPI server is running"""
        try:
            response = requests.get(f"{self.fastapi_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except Exception as e:
            print(f"⚠️  Error checking FastAPI: {e}")
            return False
    
    def wait_for_fastapi(self, max_wait=120):
        """Wait for FastAPI server to start"""
        print("=" * 70)
        print("⏳ WAITING FOR FASTAPI UI TO START")
        print("=" * 70)
        print("\n⚠️  Please start FastAPI UI manually on port 9935")
        print("🔗 FastAPI should be accessible at: http://127.0.0.1:9935")
        print("📁 Make sure FastAPI is running from:")
        print(f"   {r'your\own\path\ML Model\Scripts'}")
        print("\nThis script will wait for FastAPI to become available...")
        print("=" * 70)
        
        for i in range(max_wait):
            if self.check_fastapi_running():
                print(f"\n✅ FastAPI server is now running at {self.fastapi_url}")
                return True
            
            # Show a progress indicator every 10 seconds
            if i % 10 == 0:
                minutes = i // 60
                seconds = i % 60
                print(f"  Waiting... {minutes:02d}:{seconds:02d} | Status: Not connected")
            
            time.sleep(1)
        
        print(f"\n❌ FastAPI server did not start within {max_wait} seconds")
        print("   Please make sure:")
        print("   1. FastAPI UI is started manually")
        print("   2. It's running on port 9935")
        print("   3. The server is accessible at http://127.0.0.1:9935")
        return False
    
    def start_train_val(self):
        """Start training and validation"""
        print("\n🚀 Starting training and validation...")
        
        try:
            response = requests.post(f"{self.fastapi_url}/train_val", timeout=10)
            result = response.json()
            
            if result.get("success"):
                print("✅ Training and validation started successfully")
                print(f"   Message: {result.get('message')}")
                return True
            else:
                print(f"❌ Failed to start training: {result.get('message')}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to FastAPI at {self.fastapi_url}")
            print("   Please ensure FastAPI server is still running")
            return False
        except Exception as e:
            print(f"❌ Error starting training: {e}")
            return False
    
    def start_test(self):
        """Start testing"""
        print("\n🚀 Starting testing...")
        
        try:
            response = requests.post(f"{self.fastapi_url}/test", timeout=10)
            result = response.json()
            
            if result.get("success"):
                print("✅ Testing started successfully")
                print(f"   Message: {result.get('message')}")
                return True
            else:
                print(f"❌ Failed to start testing: {result.get('message')}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to FastAPI at {self.fastapi_url}")
            print("   Please ensure FastAPI server is still running")
            return False
        except Exception as e:
            print(f"❌ Error starting testing: {e}")
            return False
    
    def start_live_endpoint(self):
        """Start live endpoint automatically"""
        print("\n🚀 Starting live endpoint automatically...")
        
        try:
            response = requests.post(f"{self.fastapi_url}/live/start", timeout=10)
            result = response.json()
            
            if result.get("success"):
                print("✅ Live endpoint started successfully")
                print(f"   Message: {result.get('message')}")
                if result.get("data") and result["data"].get("message"):
                    print(f"   Details: {result['data']['message']}")
                return True
            else:
                print(f"❌ Failed to start live endpoint: {result.get('message')}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to FastAPI at {self.fastapi_url}")
            print("   Please ensure FastAPI server is still running")
            return False
        except Exception as e:
            print(f"❌ Error starting live endpoint: {e}")
            return False
    
    def check_live_status(self):
        """Check live endpoint status"""
        try:
            response = requests.get(f"{self.fastapi_url}/live/health", timeout=5)
            return response.json()
        except requests.exceptions.ConnectionError:
            print("⚠️  Cannot connect to FastAPI to check live endpoint status")
            return None
        except Exception as e:
            print(f"⚠️  Error checking live endpoint status: {e}")
            return None
    
    def wait_for_live_activation(self, max_wait=60):
        """Wait for live endpoint to be fully activated"""
        print("⏳ Waiting for live endpoint to activate...")
        
        start_time = time.time()
        models_loaded = False
        
        for i in range(max_wait):
            elapsed = int(time.time() - start_time)
            
            if i % 10 == 0 or elapsed % 15 == 0:
                print(f"   ⏱️  Elapsed: {elapsed}s | Checking live endpoint status...")
            
            status = self.check_live_status()
            
            if status and status.get("success"):
                data = status.get("data", {})
                
                # Check if models are loaded
                if data.get("models_loaded"):
                    loaded_models = data.get("loaded_models", [])
                    if loaded_models:
                        print(f"✅ Live endpoint activated!")
                        print(f"   Loaded models: {', '.join(loaded_models)}")
                        if data.get("next_window"):
                            print(f"   Next time window: {data['next_window']}")
                        models_loaded = True
                        return True
                    else:
                        print(f"   ⚠️  Live endpoint running but no models loaded yet...")
                else:
                    print(f"   ⚠️  Models not loaded yet...")
            
            time.sleep(2)
        
        print(f"⚠️  Live endpoint activation check timed out after {max_wait} seconds")
        print("   Note: Live endpoint may still be running, but models might not have loaded")
        return models_loaded
    
    def get_status(self):
        """Get current pipeline status"""
        try:
            response = requests.get(f"{self.fastapi_url}/status", timeout=5)
            return response.json()
        except requests.exceptions.ConnectionError:
            print("⚠️  Cannot connect to FastAPI to get status")
            return None
        except Exception as e:
            print(f"⚠️  Error getting status: {e}")
            return None
    
    def wait_for_completion(self, phase="train_val", check_interval=10):
        """Wait for a phase to complete"""
        phase_name = "Training" if phase == "train_val" else "Testing"
        print(f"\n⏳ Waiting for {phase_name} to complete...")
        
        last_symbols_count = 0
        start_time = time.time()
        last_status_time = time.time()
        
        print("   Monitoring progress (Press Ctrl+C to interrupt)...")
        
        while True:
            try:
                current_time = time.time()
                elapsed = int(current_time - start_time)
                
                # Get status every check_interval or if we need to show elapsed time
                if current_time - last_status_time >= check_interval or elapsed % 30 == 0:
                    status = self.get_status()
                    last_status_time = current_time
                    
                    if not status:
                        print(f"   ⏱️  Elapsed: {elapsed}s | Status: Cannot connect")
                        time.sleep(5)
                        continue
                    
                    if not status.get("success"):
                        print(f"   ⏱️  Elapsed: {elapsed}s | Status: Error in response")
                        time.sleep(check_interval)
                        continue
                    
                    data = status.get("data", {})
                    
                    # Check phase status
                    phase_data = data.get(phase, {})
                    running = phase_data.get("running", False)
                    phase_status = phase_data.get("status", "")
                    
                    # Get trained symbols count
                    symbols = data.get("trained_symbols", [])
                    symbols_count = len(symbols)
                    
                    # Get progress percentage if available
                    progress = phase_data.get("progress", 0)
                    
                    # Print progress
                    if symbols_count > last_symbols_count:
                        print(f"   ✅ Trained symbols: {symbols_count}")
                        if symbols:
                            print(f"      Symbols: {', '.join(symbols)}")
                        last_symbols_count = symbols_count
                    
                    # Show status every 30 seconds
                    if elapsed % 30 == 0:
                        if progress > 0:
                            print(f"   ⏱️  Elapsed: {elapsed}s | Status: {phase_status} | Progress: {progress}%")
                        else:
                            print(f"   ⏱️  Elapsed: {elapsed}s | Status: {phase_status}")
                    
                    if not running:
                        if "completed" in phase_status.lower():
                            print(f"\n✅ {phase_name} completed in {elapsed} seconds!")
                            return True
                        elif "failed" in phase_status.lower():
                            print(f"\n❌ {phase_name} failed after {elapsed} seconds: {phase_status}")
                            return False
                        elif "error" in phase_status.lower():
                            print(f"\n❌ {phase_name} errored after {elapsed} seconds: {phase_status}")
                            return False
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️  {phase_name} monitoring interrupted by user")
                choice = input(f"   Do you want to continue waiting for {phase_name}? (y/n): ")
                if choice.lower() != 'y':
                    print(f"   ⏹️  Stopped waiting for {phase_name}")
                    return False
                else:
                    print(f"   Continuing to monitor {phase_name}...")
                    continue
            except Exception as e:
                print(f"   ⚠️  Error in monitoring: {e}")
                time.sleep(check_interval)
    
    def load_parameters(self):
        """Load and display parameters"""
        try:
            with open(self.params_path, 'r') as f:
                params = json.load(f)
            
            print("\n📋 PIPELINE CONFIGURATION:")
            print("=" * 40)
            print(f"   • Epochs: {params.get('training', {}).get('num_epochs', 'N/A')}")
            print(f"   • Batch size: {params.get('training', {}).get('batch_size', 'N/A')}")
            print(f"   • Features: {', '.join(params.get('data', {}).get('features', []))}")
            print(f"   • MLflow: {'Enabled' if params.get('mlflow', {}).get('enabled') else 'Disabled'}")
            print(f"   • Auto-start live endpoint: {'Yes' if self.auto_start_live else 'No'}")
            
            # Show MLflow tracking URI if available
            mlflow_config = params.get('mlflow', {})
            if mlflow_config.get('tracking_uri'):
                print(f"   • MLflow URI: {mlflow_config.get('tracking_uri')}")
            
            print("=" * 40)
            
            return params
        except Exception as e:
            print(f"⚠️  Could not load parameters: {e}")
            return None
    
    def ask_auto_live_setting(self):
        """Ask user about auto-starting live endpoint"""
        print("\n⚙️  LIVE ENDPOINT AUTOMATION SETTING")
        print("-" * 40)
        print("Do you want the live endpoint to start automatically after testing?")
        print("  • Yes: Live endpoint will start automatically after successful testing")
        print("  • No:  You'll need to start it manually via FastAPI UI")
        
        choice = input("\nAuto-start live endpoint after testing? (y/n): ")
        if choice.lower() == 'y':
            self.auto_start_live = True
            print("✅ Live endpoint will start automatically after testing")
        else:
            self.auto_start_live = False
            print("✅ Live endpoint will NOT start automatically")
            print("   You can start it manually at: http://127.0.0.1:9935/docs")
        
        return self.auto_start_live
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("=" * 70)
        print("🤖 AUTOMATED ML PIPELINE - TRAINING, TESTING & LIVE DEPLOYMENT")
        print("=" * 70)
        print("\n⚠️  IMPORTANT: Before starting, ensure:")
        print("   1. FastAPI UI is running on http://127.0.0.1:9935")
        print("   2. MLflow UI is running on http://127.0.0.1:5000 (optional)")
        print("   3. All required data files are available")
        print("=" * 70)
        
        # Ask user about auto-starting live endpoint
        self.ask_auto_live_setting()
        
        # Load and display parameters
        params = self.load_parameters()
        if not params:
            print("❌ Cannot proceed without parameters")
            return False
        
        # Wait for FastAPI to be available
        if not self.wait_for_fastapi():
            return False
        
        print("\n" + "=" * 70)
        print("📊 PIPELINE EXECUTION STARTING")
        print("=" * 70)
        
        # Step 1: Run training and validation
        print("\n1️⃣  TRAINING & VALIDATION PHASE")
        print("-" * 40)
        
        if not self.start_train_val():
            print("❌ Training phase failed to start")
            return False
        
        if not self.wait_for_completion("train_val"):
            print("❌ Training phase failed or was interrupted")
            return False
        
        # Check trained symbols before testing
        print("\n📊 CHECKING TRAINING RESULTS")
        print("-" * 40)
        status = self.get_status()
        if status and status.get("success"):
            symbols = status.get("data", {}).get("trained_symbols", [])
            if not symbols:
                print("❌ No symbols were trained successfully")
                return False
            print(f"✅ Successfully trained {len(symbols)} symbols: {', '.join(symbols)}")
        else:
            print("⚠️  Could not verify training results, continuing anyway...")
        
        print("\n" + "=" * 70)
        
        # Step 2: Run testing
        print("\n2️⃣  TESTING PHASE")
        print("-" * 40)
        
        if not self.start_test():
            print("❌ Testing phase failed to start")
            return False
        
        if not self.wait_for_completion("test"):
            print("❌ Testing phase failed or was interrupted")
            return False
        
        print("\n" + "=" * 70)
        
        # Step 3: Automatically start live endpoint if enabled
        if self.auto_start_live:
            print("\n3️⃣  LIVE ENDPOINT DEPLOYMENT")
            print("-" * 40)
            
            print("🔄 Testing completed successfully, starting live endpoint...")
            
            if not self.start_live_endpoint():
                print("❌ Failed to start live endpoint automatically")
                print("   You can start it manually at: http://127.0.0.1:9935/docs")
            else:
                # Wait for live endpoint to be fully activated
                if self.wait_for_live_activation():
                    print("\n✅ Live endpoint is now active and monitoring time windows!")
                else:
                    print("\n⚠️  Live endpoint started but activation check timed out")
                    print("   It may still be running. Check /live/health endpoint for details.")
        
        print("\n" + "=" * 70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Final status
        print("\n📊 FINAL RESULTS SUMMARY")
        print("-" * 40)
        
        status = self.get_status()
        if status and status.get("success"):
            data = status.get("data", {})
            trained = data.get("trained_symbols", [])
            test_status = data.get("test", {})
            train_status = data.get("train_val", {})
            live_status = data.get("live", {})
            
            print(f"   • Trained symbols: {len(trained)}")
            if trained:
                print(f"     {', '.join(trained)}")
            
            print(f"   • Training status: {train_status.get('status', 'N/A')}")
            print(f"   • Testing status: {test_status.get('status', 'N/A')}")
            print(f"   • Live endpoint: {'RUNNING ✅' if live_status.get('running') else 'STOPPED ⏸️'}")
            
            # Show live endpoint details if running
            if live_status.get('running'):
                live_health = live_status.get('health', {})
                if live_health:
                    print(f"     - Models loaded: {'Yes' if live_health.get('models_loaded') else 'No'}")
                    if live_health.get('loaded_models'):
                        print(f"     - Loaded models: {', '.join(live_health['loaded_models'])}")
                    if live_health.get('next_window'):
                        print(f"     - Next time window: {live_health['next_window']}")
            
            # Show any metrics if available
            test_metrics = test_status.get('metrics', {})
            if test_metrics:
                print(f"\n   📈 TEST METRICS:")
                for key, value in test_metrics.items():
                    print(f"     • {key}: {value}")
        else:
            print("   ⚠️  Could not retrieve final status details")
        
        print("\n🔗 ACCESS YOUR RESULTS:")
        print("-" * 40)
        print("   • FastAPI Documentation: http://127.0.0.1:9935/docs")
        print("   • MLflow UI (for experiment tracking): http://127.0.0.1:5000")
        if self.auto_start_live:
            print("   • Live endpoint: Automatically started")
            print("     - Status: http://127.0.0.1:9935/live/health")
            print("     - Stop endpoint: http://127.0.0.1:9935/docs#/default/live_stop_endpoint_live_stop_post")
        else:
            print("   • Live endpoint: NOT started (manual control enabled)")
            print("     - Start manually: http://127.0.0.1:9935/docs#/default/live_start_endpoint_live_start_post")
        
        print("\n💡 TIPS:")
        print("   • Check MLflow UI for detailed experiment metrics")
        print("   • Use FastAPI /status endpoint to check current state")
        print("   • Models and logs are saved in the working directory")
        print("   • Live endpoint monitors: 7:50-8:00 and 13:50-14:00 time windows")
        print("=" * 70)
        
        # If live endpoint is running, show how to stop it
        if self.auto_start_live:
            print("\n⚠️  LIVE ENDPOINT IS CURRENTLY RUNNING")
            print("-" * 40)
            print("   To stop the live endpoint manually:")
            print("   1. Go to: http://127.0.0.1:9935/docs")
            print("   2. Find 'POST /live/stop' endpoint")
            print("   3. Click 'Try it out' then 'Execute'")
            print("\n   Or send a POST request to: http://127.0.0.1:9935/live/stop")
            print("=" * 70)
        
        return True

def main():
    """Main function"""
    automator = PipelineAutomator()
    
    try:
        print("\n⚠️  REMINDER: Make sure FastAPI UI is already running!")
        print("   You should have started it manually before running this script.")
        confirm = input("\n   Are you ready to start the pipeline? (y/n): ")
        
        if confirm.lower() != 'y':
            print("\n⏹️  Pipeline start cancelled by user")
            return 0
        
        success = automator.run_pipeline()
        
        if success:
            print("\n📱 Pipeline automation completed.")
            print("   FastAPI and MLflow UIs are still running.")
            if automator.auto_start_live:
                print("   Live endpoint is running and monitoring time windows.")
            print("   You can close them manually when finished.")
            print("=" * 70)
            
            # Optional: Keep the script running to show completion
            input("\nPress Enter to exit...")
            return 0
        else:
            print("\n❌ Pipeline failed. Check the logs above for details.")
            print("   Make sure FastAPI is still running and try again.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 Pipeline automation stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())