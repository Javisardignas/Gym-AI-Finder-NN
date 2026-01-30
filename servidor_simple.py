#!/usr/bin/env python3
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Global variables
brain = None
import_error = None

def load_brain():
    global brain, import_error
    if brain is not None:
        return True
    
    try:
        from nngym_v2 import GymBrain
        brain = GymBrain()
        if not brain.load_database(force=True):
            return False
        return True
    except Exception as e:
        import_error = str(e)
        return False

class SearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # Search endpoint
        if parsed_path.path == '/search':
            query_params = parse_qs(parsed_path.query)
            query = query_params.get('q', [''])[0].strip()
            
            if not query:
                self.send_json_response({'success': False, 'error': 'Empty query'}, 400)
                return
            
            try:
                if not load_brain():
                    raise Exception(f"Failed to load model: {import_error}")
                
                results = brain.search(query, top_k=5)
                
                formatted_results = []
                for item in results:
                    if isinstance(item, dict):
                        formatted_results.append({
                            'name': item.get('name', item.get('exercise', 'Unknown')),
                            'similarity': float(item.get('similarity', 0))
                        })
                    elif isinstance(item, tuple) and len(item) >= 2:
                        formatted_results.append({
                            'name': str(item[0]),
                            'similarity': float(item[1])
                        })
                
                self.send_json_response({
                    'success': True,
                    'results': formatted_results
                })
                
            except Exception as e:
                print(f"[ERROR] {e}")
                self.send_json_response({
                    'success': False,
                    'error': str(e)
                }, 500)
        else:
            self.send_response(404)
            self.end_headers()
    
    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        print(f"[SERVER] {format % args}")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏋️  GYM AI - FLUTTER SERVER")
    print("="*60)
    print("\n🔄 Loading database...")
    
    if load_brain():
        print("✅ Database loaded successfully!")
    else:
        print(f"⚠️  Database loading failed: {import_error}")
    
    server = HTTPServer(('localhost', 5000), SearchHandler)
    
    print("\n✅ Server active at: http://localhost:5000")
    print("📱 Flask app listening for requests from Flutter")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        server.server_close()

