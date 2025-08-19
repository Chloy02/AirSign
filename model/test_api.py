import requests
import json

def test_api_with_image(image_path: str, api_url: str = "http://localhost:8000"):
    """
    Test the ASL detection API with a local image.
    """
    try:
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post(f"{api_url}/detect-asl", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Detection successful!")
            print(f"📝 Detected word: '{result['detected_word']}'")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📝 Response: {response.text}")
            
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to API at {api_url}")
        print("💡 Make sure the server is running: uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    # Test with your existing images
    test_images = ["01.png", "02.png", "03.png"]
    
    print("🚀 Testing ASL Detection API...")
    print("=" * 50)
    
    for image in test_images:
        print(f"\n🖼️  Testing with {image}:")
        test_api_with_image(image)
    
    print("\n" + "=" * 50)
    print("✨ Testing complete!")
