"""Quick test to verify all imports work correctly."""

try:
    print("Testing imports...")
    
    from ui.main_window import MainWindow
    print("✓ MainWindow")
    
    from ui.tts_panel import TTSPanel
    print("✓ TTSPanel")
    
    from core.inference_manager import InferenceManager
    print("✓ InferenceManager")
    
    from core.backend_f5 import F5TTSBackend
    print("✓ F5TTSBackend")
    
    from ui.cloning_panel import VoiceCloningPanel
    print("✓ VoiceCloningPanel")
    
    from ui.monitor_widget import GPUMonitorWidget
    print("✓ GPUMonitorWidget")
    
    from utils.gpu_monitor import GPUMonitor
    print("✓ GPUMonitor")
    
    print("\n✓ All imports successful!")
    
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
