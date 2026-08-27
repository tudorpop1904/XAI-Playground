import requests
import io
import time

URL = 'http://127.0.0.1:8000/api/v1/analyze'

def test_analyze():
    print(f'Sending request to {URL}')
    
    # Create a dummy image
    from PIL import Image
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
    data = {
        'detector': 'CNN_Model',  # Assuming CNN_Model exists, or we could pass CNN
        'explainer': 'grad_cam',
        'enhance': 'False'
    }
    
    start = time.time()
    try:
        response = requests.post(URL, files=files, data=data)
        elapsed = time.time() - start
        
        print(f'Status Code: {response.status_code}')
        if response.status_code == 200:
            print('SUCCESS!')
            print('Response body:', response.json())
        else:
            print('FAILED!')
            print('Response body:', response.text)
            
        print(f'Elapsed time: {elapsed:.2f}s')
    except Exception as e:
        print(f'Connection failed: {e}')

if __name__ == '__main__':
    test_analyze()
